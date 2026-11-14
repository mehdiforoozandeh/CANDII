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
