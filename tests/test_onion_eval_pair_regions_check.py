"""PCk: the pair-region check of Amendment 8, section B.5, on synthetic per-region pair files.

`pair_metrics_from_records` is a test double with the pinned signature until chunk SF0 merges:
it is the scorer's `_correction_metrics` plus the replicate-pair block of `Scorer.run`, built from
the scorer's own `metric`, `_gap_fn` and `_mean_of`."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import pair_regions_check as PC  # noqa: E402
from onion_eval import select_rule as SR  # noqa: E402

RB = 16384
N_BOOT = 300          # the tool's default is 2,000; fewer draws keep the laptop test fast
ASSAYS = ("A", "B")


def pmfr_double(cor_recs, rep_recs, scorable, n_boot, seed):
    from onion_eval import metrics as Mx
    from onion_eval import score as S
    sc = scorable.get("correction", [])
    head = [r for r in cor_recs if r["never_trained"] == 1 and r["subset_pair"] == 0]
    sub = [r for r in cor_recs if r["subset_pair"] == 1]
    m = S.metric(head, S._gap_fn, sc, n_boot, seed, require_all=True)
    knobs = defaultdict(list)
    for r in head:
        knobs[r.get("knob") or ""].append(r)
    m["per_knob"] = {k: S._j(S._gap_fn(rs)) for k, rs in sorted(knobs.items())}
    m["n_pairs"] = len(head)
    den = {}
    for a in sorted(sc):
        rs = [r for r in head if r["assay"] == a]
        b, f = [r["crps_depth_only"] for r in rs], [r["crps_floor"] for r in rs]
        den[a] = (float((np.mean(b) - np.mean(f)) / np.mean(b))
                  if rs and not any(S._undef(x) for x in b + f) and np.mean(b) > 0 else None)
    m["denominator_relative_by_assay"] = den

    def law(rs):
        return Mx.depth_law_error([r["pred_total"] for r in rs], [r["source_total"] for r in rs],
                                  [r["log2_depth_ratio"] for r in rs])
    out = {
        "correction_gap_closed_vs_depth_only_never_trained": m,
        "correction_crps_model": S.metric(head, S._mean_of("crps_model"), sc, n_boot, seed),
        "correction_crps_model_oracle_scaled": S.metric(
            head, S._mean_of("crps_model_oracle_scaled"), sc, n_boot, seed),
        "correction_crps_depth_only": S.metric(head, S._mean_of("crps_depth_only"), sc, n_boot, seed),
        "correction_crps_pseudoreplicate_floor": S.metric(head, S._mean_of("crps_floor"), sc, n_boot, seed),
        "subset_pairs_gap_closed": S.metric(sub, S._gap_fn, sc, n_boot, seed),
        "depth_law_abs_log2_error_never_trained": S.metric(head, law, sc, n_boot, seed),
    }
    out["replicate_pair_gap_closed_vs_depth_only"] = {"per_rung": {
        g: S.metric([r for r in rep_recs if r["rung"] == g], S._gap_fn,
                    set((scorable.get("layers") or {}).get(g, [])), n_boot, seed)
        for g in sorted({r["rung"] for r in rep_recs})}}
    return out


# ------------------------------------------------------------------ synthetic fixture
def regions(R):
    """Region order on two chromosomes; regions 1 and 2 touch (one cluster)."""
    out, s = [], 0
    for i in range(R):
        c = "chr1" if i < R // 2 else "chr2"
        if i == R // 2:
            s = 0
        out.append([c, s])
        s += RB if i == 1 else 3 * RB
    return out


def pairs():
    cor = []
    for a in ASSAYS:
        for k in range(4):
            cor.append({"pair_id": f"c_{a}{k}", "assay": a, "cell": f"cell{k}", "knob": "lab",
                        "never_trained": 1, "subset_pair": 0, "split": "val",
                        "source": f"m_{a}{k}s", "target": f"m_{a}{k}t"})
    cor.append({"pair_id": "c_sub", "assay": "A", "cell": "cell0", "knob": "", "never_trained": 0,
                "subset_pair": 1, "split": "val", "source": "x", "target": "y"})
    rep = [{"pair_id": f"r_{k}", "assay": "A", "cell": f"cell{k}", "rung": "technical"}
           for k in range(3)]
    return cor, rep


def evalset(tmp, R):
    cor, rep = pairs()
    ev = {"regions": regions(R), "region_bins": RB, "correction_pairs": cor,
          "replicate_pairs": [r["pair_id"] for r in rep],
          "scorable": {"imputation": list(ASSAYS), "correction": list(ASSAYS),
                       "layers": {"technical": ["A"]}}}
    p = tmp / "evalset.json"
    p.write_text(json.dumps(ev))
    return p, ev


def write_run(d, ev, model_crps, *, imp, unc, ref=3.0, floor=1.0):
    """`model_crps` [R]: the model's per-bin CRPS per region (all pairs share it, plus a small
    per-pair offset); depth-only and floor are flat. Writes pairs_per_region.npz and a scores.json
    whose pair keys come from the double on all regions."""
    cor, rep = pairs()
    R = len(model_crps)
    ids = [p["pair_id"] for p in cor] + [r["pair_id"] for r in rep]
    P = len(ids)
    off = np.linspace(-0.05, 0.05, P)[:, None]
    bins = np.full((P, R), float(RB))
    src = np.full((P, R), 100.0) * (1 + 0.1 * np.arange(P)[:, None])
    arr = {"crps_sum_model": (np.asarray(model_crps)[None, :] + off) * bins,
           "crps_sum_depth_only": np.full((P, R), ref) * bins,
           "crps_sum_floor": np.full((P, R), floor) * bins, "bins": bins,
           "pred_total": src * (1.5 + 0.05 * np.arange(P)[:, None]), "source_total": src}
    meta = {p["pair_id"]: dict(p, rung="") for p in cor}
    meta.update({r["pair_id"]: dict(r, knob="", never_trained=0, subset_pair=0, source="s",
                                    target="t") for r in rep})
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "pairs_per_region.npz", pair_ids=np.array(ids), region_indices=np.arange(R),
             log2_depth_ratio=np.full(P, 0.5), **arr,
             **{k: np.array([str(meta[i][k]) for i in ids]) for k in PC.META_STR},
             **{k: np.array([int(meta[i][k]) for i in ids]) for k in PC.META_INT})
    (d / "scores.json").write_text("{}")
    run = PC.Run(d, ev)
    o = pmfr_double(*run.records({k: run.arr[k].sum(1) for k in PC.SUM_KEYS}), ev["scorable"], 0, 0)
    s = {k: o[k] for k in PC.B4_KEYS}
    s["predicted_over_replicate_variance"] = {"macro": 1.1, "per_rung": {"technical": {"macro": 0.9}}}
    s.update(split="val", regime=SR.EXPECT_REGIME, noise_floor=SR.EXPECT_NOISE_FLOOR,
             evalset_sha256="e" * 64, tables_sha256="t" * 64, tables=str(d))
    s["copy_leak"] = {"copy_leak_pass": True, "part1_wiring_pass": True}
    s["selection"] = {"space": SR.EXPECT_SPACE, "imputation_gap_closed": imp,
                      "uncertainty_gap_closed": unc, "denominators": {
                          "uncertainty_marginal_pit_deviation": 0.1}}
    s = PC.subset_scores(s, o, 0)
    del s["pair_regions_check_subset"]
    (d / "scores.json").write_text(json.dumps(s))


def build(tmp, R, base_effect, ch_effect):
    """Two arms of three seeds: base (model CRPS 2.0 + per-seed step) and scrambled (2.3 + step)."""
    p, ev = evalset(tmp, R)
    for arm, lvl, eff in (("rung1_base", 2.0, base_effect), ("rung1_scrambled_covariates", 2.3, ch_effect)):
        for s in range(3):
            write_run(tmp / arm / f"seed{s}", ev, lvl + 0.02 * s + eff,
                      imp=0.6 + 0.01 * s, unc=0.5 + 0.01 * s)
    return p


def check(tmp, p, sizes=(8, 16, 32)):
    return PC.run_check([tmp / "rung1_base", tmp / "rung1_scrambled_covariates"], tmp / "baselines",
                        p, tmp / "out", sizes=sizes, n_boot=N_BOOT, pmfr=pmfr_double,
                        log=lambda *a: None)


# ------------------------------------------------------------------ tests
def test_positions_are_nested_subsets_of_section_b4():
    pos = {p: PC.pair_region_positions(256, p) for p in (8, 16, 32)}
    assert pos[8] == list(range(0, 256, 32)) and pos[16] == list(range(0, 256, 16))
    assert pos[32] == list(range(0, 256, 8))
    assert set(pos[8]) < set(pos[16]) < set(pos[32])
    with pytest.raises(ValueError):
        PC.pair_region_positions(256, 24)


def test_adjacent_regions_form_one_cluster():
    cl = PC.clusters([tuple(r) for r in regions(64)], RB)
    assert cl[1] == cl[2] and cl[0] != cl[1] and cl[3] != cl[2]
    assert cl.max() + 1 == 63
    W = PC.boot_weights(cl, 50)
    assert np.array_equal(W[:, 1], W[:, 2]) and np.allclose(W.sum(1), 63 + W[:, 1])


def test_uniform_regions_pass_at_8(tmp_path):
    p = build(tmp_path, 64, np.zeros(64), np.zeros(64))
    res = check(tmp_path, p)
    assert all(res["summary"][8].values()), res["summary"]
    assert res["chosen"] == 8
    first = (tmp_path / "out" / "PAIR_REGIONS.md").read_text().splitlines()[0]
    assert first == "pair regions: 8"
    tsv = (tmp_path / "out" / "check.tsv").read_text().splitlines()
    assert tsv[0].startswith("p\tcondition\t") and len(tsv) > 10


def planted_region_effect(R, D=0.5, s=0.05, seed=3):
    """+D on the 8-subset, -D/3 on the rest of the 32-subset (so its mean is 0), small noise on
    the regions outside the 32-subset."""
    e = np.random.default_rng(seed).normal(0, s, R)
    step8, step32 = R // 8, R // 32
    for i in range(0, R, step32):
        e[i] = D if i % step8 == 0 else -D / 3
    return e


def test_region_effect_fails_at_8_passes_at_32(tmp_path):
    e = planted_region_effect(64)
    p = build(tmp_path, 64, e, e)
    res = check(tmp_path, p)
    assert not res["summary"][8]["1"] and not res["summary"][16]["1"]
    assert all(res["summary"][32].values()), res["summary"][32]
    assert res["chosen"] == 32


def test_verdict_flip_fails_condition_3(tmp_path):
    R = 64
    ch = np.zeros(R)
    ch[::R // 8] = -0.6        # the challenger is much better on the 8-subset regions only
    p = build(tmp_path, R, np.zeros(R), ch)
    res = check(tmp_path, p, sizes=(8,))
    assert res["summary"][8]["3"] is False
    row = [r for r in res["rows"] if r["p"] == 8 and r["condition"] == "3" and r["scope"] != "ALL"][0]
    assert row["value_all_regions"] == "INCUMBENT" and row["value"] != "INCUMBENT"
    assert (tmp_path / "out" / "PAIR_REGIONS.md").read_text().startswith(
        "pair regions: 64 (none of 8 passes)")


def test_subset_copy_moves_only_b4_keys_and_keeps_metric7(tmp_path):
    p = build(tmp_path, 64, np.zeros(64), np.zeros(64))
    check(tmp_path, p, sizes=(16,))
    a = json.loads((tmp_path / "rung1_base" / "seed0" / "scores.json").read_text())
    b = json.loads((tmp_path / "out" / "subset_scores" / "p16" / "rung1_base" / "seed0" /
                    "scores.json").read_text())
    moved = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert moved <= set(PC.B4_KEYS) | {"selection", "pair_regions_check_subset"}
    assert a["predicted_over_replicate_variance"] == b["predicted_over_replicate_variance"]


def test_thresholds_equal_the_preregistration():
    assert PC.SIZES == (8, 16, 32) and PC.N_BOOT == 2000 and PC.SPREAD_RATIO_MAX == 1.5
    assert PC.PER_ASSAY_SHARE_MIN == 0.95 and PC.INTERVAL == (2.5, 97.5)
    pre = list((REPO / "cruxvault" / "results").glob("*/PAIR_CHECK_PREREGISTRATION.md"))
    if not pre:
        pytest.skip("no PAIR_CHECK_PREREGISTRATION.md in this checkout")
    m = re.search(r"^Constants: `(\{.*\})`$", pre[0].read_text(), re.M)
    assert m and json.loads(m.group(1)) == PC.thresholds()


# ------------------------------------------------------------------ the amended check (2026-11-16)
NND = {"0": 0, "1": 0, "2": 1, "3": 1}      # pairs k = 0, 1 have no duplicate-keeping side (crop pairs)
SV_GAIN = 0.3                        # the same-view CRPS is the cross-view CRPS minus this


def write_oracle(d, ev, R, cols=None, o=0.5, extra_assay=None):
    """An oracle reference (oracle_floor.load_oracle form) for the headline correction pairs;
    `extra_assay` adds one crop pair of that assay that no run scores."""
    head, _ = pairs()                # every correction pair, as the scorer's reference holds
    if extra_assay:
        head = head + [{"pair_id": f"c_{extra_assay}0", "assay": extra_assay, "cell": "cellX",
                        "knob": "lab", "never_trained": 1, "subset_pair": 0, "source": "xs",
                        "target": "xt"}]
    ri = np.arange(R) if cols is None else np.asarray(cols)
    n = len(head)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "oracle_per_region.npz", pair_ids=np.array([p["pair_id"] for p in head]),
             region_indices=ri, crps_sum_oracle=np.full((n, len(ri)), o * RB),
             bins=np.full((n, len(ri)), float(RB)),
             **{k: np.zeros((n, len(ri))) for k in ("sq_half_sum", "pr1_sum", "pr2_sum")},
             assay=np.array([p["assay"] for p in head]), cell=np.array([p["cell"] for p in head]),
             rung=np.array([""] * n), knob=np.array([p["knob"] for p in head]),
             source=np.array([p["source"] for p in head]), target=np.array([p["target"] for p in head]),
             oracle_reason=np.array([""] * n), never_trained=np.array([p["never_trained"] for p in head]),
             subset_pair=np.array([p["subset_pair"] for p in head]), source_dedup=np.ones(n, int),
             target_dedup=np.ones(n, int),
             nodedup_side=np.array([NND.get(p["pair_id"][-1], 0) for p in head]))
    (d / "oracle_floor.json").write_text(json.dumps({"evalset_sha256": "e" * 64,
                                                     "tables_sha256": "t" * 64}))


def add_v3(d, ev, model_crps, gain=SV_GAIN, refs_dir=None):
    """Same-view file, scorable and the correction-only term on a write_run seed directory."""
    from onion_eval import score as S
    cor, _ = pairs()
    targets = [p["target"] for p in cor if p["never_trained"] == 1 and p["subset_pair"] == 0]
    R = len(model_crps)
    np.savez(d / "same_view_per_region.npz", pair_ids=np.array(["same_view/" + t for t in targets]),
             region_indices=np.arange(R), target=np.array(targets),
             assay=np.array([t[2] for t in targets]), cell=np.array(["c"] * len(targets)),
             crps_sum_model=np.tile((np.asarray(model_crps) - gain) * RB, (len(targets), 1)),
             bins=np.full((len(targets), R), float(RB)),
             crps_model_same_view=np.full(len(targets), float(np.mean(model_crps)) - gain))
    s = json.loads((d / "scores.json").read_text())
    s["scorable"] = ev["scorable"]
    run = PC.Run(d, ev)
    cr, _ = run.records({k: run.arr[k].sum(1) for k in PC.SUM_KEYS})
    s[S.CORRECTION_ONLY_KEY] = S.correction_only_block(
        cr, [{"target": t, "crps_model_same_view": float(np.mean(model_crps)) - gain} for t in targets])
    if refs_dir is not None:
        s["refs"] = {"dir": str(refs_dir)}
    (d / "scores.json").write_text(json.dumps(s))


def screening_copy(src, dst, ev, p, refs_dir):
    """The p-region screening score of a seed: the npz columns of the subset, as a p16_sv dir."""
    from onion_eval import score as S
    pos = PC.pair_region_positions(len(ev["regions"]), p)
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("pairs_per_region.npz", "same_view_per_region.npz"):
        z = dict(np.load(src / f))
        R = len(z["region_indices"])
        for k, v in list(z.items()):
            if v.ndim == 2 and v.shape[1] == R:
                z[k] = v[:, pos]
        z["region_indices"] = np.asarray(pos)
        np.savez(dst / f, **z)
    s = json.loads((src / "scores.json").read_text())
    s["refs"] = {"dir": str(refs_dir)}
    (dst / "scores.json").write_text(json.dumps(s))
    run = PC.Run(dst, ev)
    cr, _ = run.records({k: run.arr[k].sum(1) for k in PC.SUM_KEYS})
    z = np.load(dst / "same_view_per_region.npz")
    sv = [{"target": str(t), "crps_model_same_view": float(c.sum() / b.sum())}
          for t, c, b in zip(z["target"], z["crps_sum_model"], z["bins"])]
    s[S.CORRECTION_ONLY_KEY] = S.correction_only_block(cr, sv)
    (dst / "scores.json").write_text(json.dumps(s))


D_B = 4.0                            # assay B: depth-only CRPS 4.0 (assay A: 3.0)
SHIFT_B, SHIFT_NND = 0.4, 0.8        # model CRPS added to assay B pairs and to duplicate-keeping pairs


def distinguish(d, assay_eff=None):
    """Make pairs differ: assay B has another depth-only CRPS and model level, the duplicate-keeping
    pairs (k = 2, 3) a much worse model; `assay_eff` {assay: [R]} adds a per-region model effect."""
    z = dict(np.load(d / "pairs_per_region.npz"))
    b = z["bins"]
    for i, pid in enumerate(z["pair_ids"]):
        pid, a = str(pid), str(z["assay"][i])
        if not pid.startswith("c_"):
            continue
        if a == "B":
            z["crps_sum_depth_only"][i] *= D_B / 3.0
            z["crps_sum_model"][i] += SHIFT_B * b[i]
        if pid[-1] in "23":
            z["crps_sum_model"][i] += SHIFT_NND * b[i]
        if assay_eff and a in assay_eff:
            z["crps_sum_model"][i] += np.asarray(assay_eff[a]) * b[i]
    np.savez(d / "pairs_per_region.npz", **z)


def expected_v3(d, o=0.5):
    """The version-3 term computed here from the npz: crop pairs (k = 0, 1) only, per assay."""
    z = np.load(d / "pairs_per_region.npz")
    out = []
    for a in ASSAYS:
        rows = [i for i, pid in enumerate(z["pair_ids"]) if str(pid) in (f"c_{a}0", f"c_{a}1")]
        m = np.mean([z["crps_sum_model"][i].sum() / z["bins"][i].sum() for i in rows])
        dd = np.mean([z["crps_sum_depth_only"][i].sum() / z["bins"][i].sum() for i in rows])
        out.append((dd - m) / (dd - o))
    return float(np.mean(out))


def build_v3(tmp, R, effects, screening=True, assay_eff=None, extra_assay=None):
    """Arms named in `effects` {arm: (level, region effect [R] or f(seed) -> [R])}, 3 seeds,
    scored into p256_sv, with the distinguished pairs."""
    p, ev = evalset(tmp, R)
    write_oracle(tmp / "oracle_floor" / "seed0", ev, R, extra_assay=extra_assay)
    write_oracle(tmp / "baselines_p16" / "oracle_floor" / "seed0", ev, R,
                 cols=PC.pair_region_positions(R, 16), extra_assay=extra_assay)
    for arm, (lvl, eff) in effects.items():
        for s in range(3):
            m = lvl + 0.02 * s + (eff(s) if callable(eff) else eff)
            d = tmp / arm / f"seed{s}" / "p256_sv"
            write_run(d, ev, m, imp=0.6 + 0.01 * s, unc=0.5 + 0.01 * s)
            distinguish(d, (assay_eff or {}).get(arm))
            add_v3(d, ev, m, gain=SV_GAIN + 0.01 * s)      # seeds differ in the term
            if screening:
                screening_copy(d, tmp / arm / f"seed{s}" / "p16_sv", ev, 16, tmp / "baselines_p16")
    return p


def check_v3(tmp, p, arms, comparisons, sizes=(8, 16, 32)):
    return PC.run_check_v3([tmp / a for a in arms], p, tmp / "oracle_floor" / "seed0", tmp / "out",
                           screening_sub="p16_sv", comparisons=comparisons, sizes=sizes,
                           n_boot=N_BOOT, pmfr=pmfr_double, log=lambda *a: None)


def test_v3_uniform_regions_pass_and_settle(tmp_path):
    R = 64
    arms = {"armA": (2.0, np.zeros(R)), "armB": (2.3, np.zeros(R))}
    p = build_v3(tmp_path, R, arms)
    res = check_v3(tmp_path, p, list(arms), (("armA", "armB"),))
    assert all(res["summary"][8].values()), res["summary"]
    assert res["chosen"] == 8 and res["screening_settled"] is True
    lines = (tmp_path / "out" / "PAIR_REGIONS_V3.md").read_text().splitlines()
    assert lines[0] == "pair regions: 8" and lines[1] == "16-region screening scores settled: yes"
    cons = [r for r in res["rows"] if r["condition"] == "consistency"]
    assert cons and all(r["value"] != "" and float(r["value"]) < 1e-9 for r in cons), cons
    # the version-3 term uses only the crop pairs, per assay (computed here from the files)
    v3 = [r for r in res["rows"] if r["condition"] == "1" and r["quantity"] == "correction_v3"
          and r["scope"] == "armA/seed0"]
    assert v3 and abs(float(v3[0]["value_all_regions"])
                      - expected_v3(tmp_path / "armA" / "seed0" / "p256_sv")) < 1e-9
    rank = [r for r in res["rows"] if r["condition"] == "6" and r["scope"] != "ALL"]
    assert rank and all(r["pass"] == "pass" for r in rank if r["pass"])


def test_v3_region_effect_fails_at_8_passes_at_32(tmp_path):
    R = 64
    e = planted_region_effect(R)
    arms = {"armA": (2.0, e), "armB": (2.3, e)}
    p = build_v3(tmp_path, R, arms, screening=False)
    res = check_v3(tmp_path, p, list(arms), (("armA", "armB"),))
    assert not res["summary"][8]["1"]
    assert all(res["summary"][32].values()), res["summary"][32]
    assert res["chosen"] == 32 and res["screening_settled"] is False


def test_v3_rank_flip_fails_condition_6(tmp_path):
    R = 64
    ch = np.zeros(R)
    ch[::R // 8] = -0.6        # armB beats armA on the 8-subset only; it is worse on all regions
    arms = {"armA": (2.0, np.zeros(R)), "armB": (2.3, ch)}
    p = build_v3(tmp_path, R, arms, screening=False)
    res = check_v3(tmp_path, p, list(arms), (("armA", "armB"),), sizes=(8,))
    assert res["summary"][8]["6"] is False and res["chosen"] is None
    flip = [r for r in res["rows"] if r["condition"] == "6" and r["pass"] == "fail"]
    assert flip and float(flip[0]["value"]) * float(flip[0]["value_all_regions"]) < 0


def test_v3_off_leaves_the_original_check_unchanged():
    assert PC.CONDITIONS == ("1", "2", "3", "4", "5")
    t = PC.thresholds()
    assert "amended" not in t and set(PC.thresholds_v3()) > set(t)
    assert {k: PC.thresholds_v3()[k] for k in t} == t


def test_thresholds_v3_equal_the_amended_preregistration():
    pre = list((REPO / "cruxvault" / "results").glob("*/PAIR_CHECK_PREREGISTRATION.md"))
    if not pre:
        pytest.skip("no PAIR_CHECK_PREREGISTRATION.md in this checkout")
    m = re.search(r"^Amended constants: `(\{.*\})`$", pre[0].read_text(), re.M)
    if not m:
        pytest.skip("the preregistration in this checkout has no amendment")
    assert json.loads(m.group(1)) == PC.thresholds_v3()


SUB8 = lambda R: np.isin(np.arange(R), PC.pair_region_positions(R, 8)).astype(float)  # noqa: E731


def test_v3_term_takes_crop_pairs_per_assay():
    """A wrong pair subset (duplicate-keeping pairs in) or a pooled assay grouping moves the term."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    build_v3(tmp, 32, {"armA": (2.0, np.zeros(32))}, screening=False)
    d = tmp / "armA" / "seed0" / "p256_sv"
    ev = json.loads((tmp / "evalset.json").read_text())
    from onion_eval import oracle_floor as OF
    r = PC.RunV3(d, ev, OF.load_oracle(tmp / "oracle_floor" / "seed0"))
    t = r.terms(np.ones(32), pmfr_double, ev["scorable"])
    assert abs(t["v3"] - expected_v3(d)) < 1e-12
    assert len(r.v3_rows) == 4 and set(t["v3_assay"]) == {"A", "B"}
    assert abs(t["v3_assay"]["A"] - t["v3_assay"]["B"]) > 0.01      # assays differ in the fixture


def test_v3_missing_assay_makes_the_term_undefined_and_fails(tmp_path):
    R = 64
    p = build_v3(tmp_path, R, {"armA": (2.0, np.zeros(R)), "armB": (2.3, np.zeros(R))},
                 screening=False, extra_assay="C")
    for d in tmp_path.glob("arm*/seed*/p256_sv"):
        s = json.loads((d / "scores.json").read_text())
        s["scorable"] = dict(s["scorable"], correction=list(ASSAYS) + ["C"])
        (d / "scores.json").write_text(json.dumps(s))
    ev = json.loads(p.read_text())
    from onion_eval import oracle_floor as OF
    r = PC.RunV3(tmp_path / "armA" / "seed0" / "p256_sv", ev, OF.load_oracle(tmp_path / "oracle_floor" / "seed0"))
    t = r.terms(np.ones(R), pmfr_double, ev["scorable"])
    assert t["v3"] is None and "C" in t["v3_reason"]
    res = check_v3(tmp_path, p, ["armA", "armB"], (("armA", "armB"),), sizes=(8,))
    assert res["summary"][8]["1"] is False and res["summary"][8]["2"] is False


def test_v3_per_assay_effect_fails_condition_2(tmp_path):
    R = 64
    X = 0.3        # per-assay ratio moves +-X on the 8-subset regions; the macro does not move
    eff = {"A": X * 2.5 * SUB8(R), "B": -X * 3.5 * SUB8(R)}
    p = build_v3(tmp_path, R, {"armA": (2.0, np.zeros(R)), "armB": (2.3, np.zeros(R))},
                 screening=False, assay_eff={"armA": eff, "armB": eff})
    res = check_v3(tmp_path, p, ["armA", "armB"], (("armA", "armB"),), sizes=(8,))
    assert res["summary"][8]["2"] is False


def test_v3_verdict_flip_fails_condition_3(tmp_path):
    R = 64
    p = build_v3(tmp_path, R, {"armA": (2.0, np.zeros(R)), "armB": (2.6, -1.2 * SUB8(R))},
                 screening=False)
    res = check_v3(tmp_path, p, ["armA", "armB"], (("armA", "armB"),), sizes=(8,))
    assert res["verdicts_256"][("armA", "armB")][0] == "INCUMBENT"
    assert res["summary"][8]["3"] is False


def test_v3_seed_spread_on_subset_fails_condition_4(tmp_path):
    R = 64
    p = build_v3(tmp_path, R, {"armA": (2.0, lambda s: 0.3 * s * SUB8(R)), "armB": (2.3, np.zeros(R))},
                 screening=False)
    res = check_v3(tmp_path, p, ["armA", "armB"], (("armA", "armB"),), sizes=(8,))
    assert res["summary"][8]["4"] is False


@pytest.mark.parametrize("eps", [1e-3, 8e-7])     # 8e-7: inside 1e-6 absolute, outside 1e-6 relative
def test_v3_stored_scores_that_differ_fail_condition_s(tmp_path, eps):
    R = 64
    arms = {"armA": (2.0, np.zeros(R)), "armB": (2.3, np.zeros(R))}
    p = build_v3(tmp_path, R, arms)
    f = tmp_path / "armA" / "seed0" / "p16_sv" / "pairs_per_region.npz"
    z = dict(np.load(f))
    z["crps_sum_model"] = z["crps_sum_model"] * (1 + eps)
    np.savez(f, **z)
    res = check_v3(tmp_path, p, list(arms), (("armA", "armB"),), sizes=(16,))
    assert all(res["summary"][16].values()) and res["screening_settled"] is False
    assert [r for r in res["rows"] if r["condition"] == "S" and r["pass"] == "fail"]


def test_v3_refuses_inconsistent_terms_other_oracle_and_seed_sets(tmp_path):
    import shutil
    R = 64
    arms = {"armA": (2.0, np.zeros(R)), "armB": (2.3, np.zeros(R))}
    p = build_v3(tmp_path, R, arms, screening=False)
    shutil.copytree(tmp_path / "oracle_floor", tmp_path / "other" / "oracle_floor")
    with pytest.raises(SystemExit, match="oracle reference"):
        PC.run_check_v3([tmp_path / a for a in arms], p, tmp_path / "other" / "oracle_floor" / "seed0",
                        tmp_path / "out", comparisons=(("armA", "armB"),), sizes=(8,), n_boot=20,
                        pmfr=pmfr_double, log=lambda *a: None)
    sj = tmp_path / "armA" / "seed1" / "p256_sv" / "scores.json"
    s = json.loads(sj.read_text())
    s["correction_only_covariate_cost_closed"]["value"] += 1e-6
    sj.write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="refuses"):
        check_v3(tmp_path, p, list(arms), (("armA", "armB"),), sizes=(8,))
    s["correction_only_covariate_cost_closed"]["value"] -= 1e-6
    sj.write_text(json.dumps(s))
    (tmp_path / "armB" / "seed3").mkdir()
    with pytest.raises(SystemExit, match="seeds"):
        check_v3(tmp_path, p, list(arms), (("armA", "armB"),), sizes=(8,))
    (tmp_path / "armB" / "seed3").rmdir()
    (tmp_path / "armB" / "seed2" / "CAPPED").write_text("")
    with pytest.raises(SystemExit, match="capped"):
        check_v3(tmp_path, p, list(arms), (("armA", "armB"),), sizes=(8,))


def test_v3_condition_2_needs_72_cells_on_the_registered_arms(tmp_path):
    R = 64
    names = [x for ab in PC.COMPARISONS_V3 for x in ab]
    arms = {a: (2.0 + 0.1 * i, np.zeros(R)) for i, a in enumerate(dict.fromkeys(names))}
    p = build_v3(tmp_path, R, arms, screening=False)
    res = check_v3(tmp_path, p, list(arms), PC.COMPARISONS_V3, sizes=(8,))
    row = [r for r in res["rows"] if r["condition"] == "2" and r["scope"] == "ALL"][0]
    assert res["summary"][8]["2"] is False and "72" in row["note"]     # 4 x 3 x 2 = 24 cells


def test_v3_bootstrap_values_drop_the_aggregation(tmp_path, monkeypatch):
    R = 64
    arms = {"armA": (2.0, np.zeros(R)), "armB": (2.3, np.zeros(R))}
    p = build_v3(tmp_path, R, arms, screening=False)
    kept, orig = [], PC.RunV3.terms

    def spy(self, *a, **k):
        v = orig(self, *a, **k)
        kept.append(v)
        return v
    monkeypatch.setattr(PC.RunV3, "terms", spy)
    check_v3(tmp_path, p, list(arms), (("armA", "armB"),), sizes=(8,))
    # only the 6 subset evaluations (p = 8) keep the aggregation; all-region and bootstrap ones drop it
    assert sum("_out" in v for v in kept) == 6 and len(kept) == 6 * (1 + N_BOOT + 1)

