"""SF0 (Amendment 8): the legacy scorer's pair rules, pair regions and per-region pair sums.

Pins, on synthetic tables with planted replicate pairs:
  - a replicate pair whose members are in two cells, and one whose source is an evaluation-only
    track, are no pair job; the scoring ends; `pair_jobs_not_applicable` gives the counts and the
    reasons; a rung left with no pair job prints Undefined with the reason; metric 7 per rung and the
    explained fraction per layer still read the cross-cell pairs (from the counts and the items);
  - every other scores.json key and every suffstats array equal a scoring of the same eval set
    without the planted pairs (the composite reads correction pairs only, so it cannot move);
  - `pairs_per_region.npz`: the per-region sums add up to each pair's totals;
  - `pair_regions`: only the pair keys of section B.4 move, and a pair's CRPS equals its CRPS on an
    eval set that holds only those regions;
  - `--pairs-only` gives the pair records and pair metrics of a full scoring, and the same file.
The identity with 9cd5f04 on the unplanted case is pinned by test_onion_eval_score_memory.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_onion_eval_ag2_dump as T  # noqa: E402
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import evalset as E  # noqa: E402
from onion_eval import score as S  # noqa: E402

RB = 256
N_REGIONS = 4
KW = dict(n_boot=10, guardrail_windows=4, log=lambda *a: None)
EO_ID = "EVALONLY_SF0.s0.H3K4me3"
#: the pair keys of Amendment 8 section B.4 (the only keys that pair regions may move)
PAIR_KEYS = ("correction_gap_closed_vs_depth_only_never_trained", "correction_crps_model",
             "correction_crps_model_oracle_scaled", "correction_crps_depth_only",
             "correction_crps_pseudoreplicate_floor", "subset_pairs_gap_closed",
             "depth_law_abs_log2_error_never_trained", "replicate_pair_gap_closed_vs_depth_only")
VOLATILE = T.VOLATILE + ("evalset",)


def _build(root: Path, plant: bool) -> dict:
    t = make_synthetic_tables(root, n_cells=9, val_tracks=T.VAL, blind_tracks=T.BLIND)
    rows = T._tsv(t / "measurements.tsv")
    base = {(r["cell"], r["assay"]): r for r in rows if r["arm_knob"] == ""}
    h = {c: base[(c, "H3K4me3")]["measurement_id"] for c in ("s0", "s1", "s2")}
    pairs = T._tsv(t / "pairs.tsv")
    arm = next(p["b"] for p in pairs if p["a"] == h["s0"] and p["split"] == "val"
               and p["subset_pair"] == "0")
    planted = []
    if plant:
        eo = dict(base[("s0", "H3K4me3")], measurement_id=EO_ID, in_eic="evalonly")
        rows.append(eo)
        T._write_tsv(t / "measurements.tsv", rows)
        tmpl = dict(pairs[0], knobs_differing="", subset_pair="0", split="val", never_trained="0",
                    assay="H3K4me3")
        planted = [dict(tmpl, pair_id="pX1", a=h["s0"], b=h["s1"], rung="other_cell_type", cell="s0"),
                   dict(tmpl, pair_id="pX2", a=h["s1"], b=h["s2"], rung="other_cell_type", cell="s1"),
                   dict(tmpl, pair_id="pE1", a=EO_ID, b=h["s0"], rung="anisogenic", cell="s0")]
    # one same-cell replicate pair that stays a pair job, in both builds
    keep = dict(pairs[0], pair_id="pI1", a=h["s0"], b=arm, rung="isogenic", cell="s0",
                assay="H3K4me3", knobs_differing="", subset_pair="0", split="val",
                never_trained="0")
    T._write_tsv(t / "pairs.tsv", pairs + [keep] + planted)
    ev = root / "eval" / "synth" / "evalset.json"
    ev.parent.mkdir(parents=True, exist_ok=True)
    d = E.build(t, ev, n_regions=N_REGIONS, n_fit_regions=1, region_bins=RB, seed=0)
    d["imputation_targets"] = [x for x in d["imputation_targets"]
                               if x["assay"] in ("H3K4me3", "DNase-seq") and x["target"] != EO_ID]
    d["correction_pairs"] = d["correction_pairs"][:3]
    d["replicate_pairs"] = ["pI1"] + [p["pair_id"] for p in planted]
    d["scorable"]["imputation"] = ["DNase-seq", "H3K4me3"]
    d["scorable"].setdefault("layers", {})["isogenic"] = ["H3K4me3"]
    ev.write_text(json.dumps(d))
    return {"root": root, "tables": t, "evalset": ev, "h": h}


def _score(env, out, evalset=None, **kw):
    return S.score(evalset or env["evalset"], out, baseline="marginal_nb", config=dict(T.CFG),
                   **KW, **kw)


def _flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _flat(v, f"{p}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _flat(v, f"{p}[{i}]")
    else:
        yield p, json.dumps(x)


def _load(out: Path) -> dict:
    s = json.loads((out / "scores.json").read_text())
    for k in VOLATILE:
        s.pop(k, None)
    return s


def _suff(out: Path) -> dict:
    with np.load(out / "suffstats.npz") as z:
        return {k: z[k].copy() for k in z.files}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("sf0")
    planted, plain = _build(root / "planted", True), _build(root / "plain", False)
    _score(planted, root / "out_planted")
    _score(plain, root / "out_plain")
    return {"root": root, "planted": planted, "plain": plain,
            "out_planted": root / "out_planted", "out_plain": root / "out_plain"}


def test_cross_cell_and_evalonly_pairs_are_no_pair_jobs(runs):
    s = _load(runs["out_planted"])
    na = s["pair_jobs_not_applicable"]
    assert na["other_cell_type"] == 2 and na["anisogenic"] == 1
    assert "different cells" in na["reasons"]["other_cell_type"]
    assert "evaluation-only" in na["reasons"]["anisogenic"]
    assert s["n_replicate_pairs"] == 1                          # pI1 only
    rows = s["replicate_pair_gap_closed_vs_depth_only"]["per_rung"]
    for rung in ("other_cell_type", "anisogenic"):
        assert rows[rung]["macro"] is None and rows[rung]["reason"].startswith("no pair job")
    assert "isogenic" in rows and rows["isogenic"]["n"] == 1
    with np.load(runs["out_planted"] / "pairs_per_region.npz") as z:
        assert not any(p in set(z["pair_ids"]) for p in ("pX1", "pX2", "pE1"))
        assert "pI1" in set(z["pair_ids"])


def test_metric7_and_explained_fraction_still_read_cross_cell_pairs(runs):
    s = _load(runs["out_planted"])
    t0 = runs["planted"]["h"]["s0"]
    h7 = s["predicted_over_replicate_variance"]["per_target"][t0]["per_rung"]
    assert "other_cell_type" in h7
    ef = json.dumps(s["explained_fraction_per_layer"])
    assert "other_cell_type" in ef
    p = _load(runs["out_plain"])
    assert "other_cell_type" not in p["predicted_over_replicate_variance"]["per_target"][t0]["per_rung"]


def test_only_listed_keys_change_and_composite_keeps_its_data(runs):
    """Planted against plain: the keys that differ are the new key, the two Undefined per-rung rows,
    and the readers of the cross-cell pairs (metric 7 per rung, the explained fraction); the
    selection block, headline 4 and 5, every correction block and every suffstats array are
    equal."""
    a, b = dict(_flat(_load(runs["out_plain"]))), dict(_flat(_load(runs["out_planted"])))
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    allowed = ("/pair_jobs_not_applicable", "/replicate_pair_gap_closed_vs_depth_only/per_rung/",
               "/predicted_over_replicate_variance", "/explained_fraction_per_layer",
               "/accessibility_only/predicted_over_replicate_variance",
               "/accessibility_only/explained_fraction_per_layer", "/tables_sha256", "/tables")
    bad = sorted(k for k in changed if not k.startswith(allowed))
    assert not bad, bad[:10]
    assert not any(k.startswith("/selection") for k in changed)
    sa, sb = _suff(runs["out_plain"]), _suff(runs["out_planted"])
    assert set(sa) == set(sb)
    for k in sa:
        np.testing.assert_array_equal(sa[k], sb[k])


def test_per_region_sums_add_up(runs):
    s = json.loads((runs["out_planted"] / "scores.json").read_text())
    with np.load(runs["out_planted"] / "pairs_per_region.npz") as z:
        ids = list(z["pair_ids"])
        assert list(z["region_indices"]) == list(range(N_REGIONS))
        assert z["bins"].shape == (len(ids), N_REGIONS) and (z["bins"] == RB).all()
        suff = _suff(runs["out_planted"])
        for i, pid in enumerate(ids):
            crps_m, crps_d, crps_f, nbins = suff[f"pair/{pid}"]
            tot = z["bins"][i].sum()
            assert tot == nbins
            for col, ref in (("crps_sum_model", crps_m), ("crps_sum_depth_only", crps_d),
                             ("crps_sum_floor", crps_f)):
                if np.isfinite(ref):
                    assert z[col][i].sum() == pytest.approx(ref * tot, rel=1e-9)
            assert np.isfinite(z["source_total"][i]).all()
    assert s["scorer_version"] == 1


def test_pair_regions_move_only_pair_keys(runs, tmp_path):
    env = runs["planted"]
    d = json.loads(env["evalset"].read_text())
    sub = [1, 3]
    d1 = dict(d, pair_regions=sub)
    e1 = tmp_path / "eval" / "x" / "evalset_pr.json"
    e1.parent.mkdir(parents=True)
    e1.write_text(json.dumps(dict(d1, tables_dir=str(env["tables"]))))
    _score(env, tmp_path / "pr", evalset=e1)
    a, b = _load(runs["out_planted"]), _load(tmp_path / "pr")
    assert b["pair_regions_n"] == 2
    fa, fb = dict(_flat(a)), dict(_flat(b))
    changed = {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}
    bad = sorted(k for k in changed if not k.startswith(tuple("/" + k for k in PAIR_KEYS))
                 and k not in ("/pair_regions_n", "/selection/correction_gap_closed",
                               "/selection/composite", "/selection/denominators/"
                               "correction_baseline_minus_floor_relative_min")
                 and not k.startswith("/selection/denominators/correction_baseline_minus_floor"))
    assert not bad, bad[:10]
    # a pair's CRPS on the pair regions equals its CRPS on an eval set of only those regions
    d2 = dict(d, regions=[d["regions"][i] for i in sub], tables_dir=str(env["tables"]))
    e2 = tmp_path / "eval" / "y" / "evalset_only.json"
    e2.parent.mkdir(parents=True)
    e2.write_text(json.dumps(d2))
    _score(env, tmp_path / "only", evalset=e2)
    s1, s2 = _suff(tmp_path / "pr"), _suff(tmp_path / "only")
    pairs = [k for k in s1 if k.startswith("pair/")]
    assert pairs
    for k in pairs:
        np.testing.assert_array_equal(s1[k], s2[k])


def test_pairs_only_equals_full_scoring(runs, tmp_path):
    env = runs["planted"]
    res = _score(env, tmp_path / "po", pairs_only=True)
    assert not (tmp_path / "po" / "scores.json").exists()
    full = _load(runs["out_planted"])
    po = json.loads((tmp_path / "po" / "pairs_only.json").read_text())
    for k in PAIR_KEYS:
        assert json.dumps(po[k], sort_keys=True) == json.dumps(full[k], sort_keys=True), k
    assert po["pair_jobs_not_applicable"] == full["pair_jobs_not_applicable"]
    assert res["n_pairs"] == full["n_pairs"] and res["n_replicate_pairs"] == full["n_replicate_pairs"]
    with np.load(tmp_path / "po" / "pairs_per_region.npz") as a, \
            np.load(runs["out_planted"] / "pairs_per_region.npz") as b:
        assert sorted(a.files) == sorted(b.files)
        for k in a.files:
            np.testing.assert_array_equal(a[k], b[k])


def test_one_subset_rule_for_derive_scorer_and_check_tool(runs, tmp_path):
    """The positions of a pair-region subset come from one function (`evalset.pair_positions`):
    `evalset.py derive`, the scorer's `pair_regions` and the check tool agree on a toy eval set."""
    from onion_eval import pair_regions_check as PC
    env = runs["planted"]
    out = tmp_path / "eval" / "z" / "evalset_p2.json"
    out.parent.mkdir(parents=True)
    E.derive(env["evalset"], out, pair_regions=2)
    d = json.loads(out.read_text())
    want = E.pair_positions(N_REGIONS, 2)
    assert d["pair_regions"] == want == PC.pair_region_positions(N_REGIONS, 2) == [0, 2]
    d["tables_dir"] = str(env["tables"])
    out.write_text(json.dumps(d))
    sc = S.Scorer(out, main="marginal_nb", config=dict(T.CFG), **KW)
    assert sc.pair_region_idx == want


def test_check_tool_reads_the_pair_file(runs):
    """pairs_per_region.npz carries what `pair_regions_check.Run` needs (log2_depth_ratio and the
    per-pair metadata), and the records it rebuilds from all regions give the scorer's pair keys."""
    from onion_eval import pair_regions_check as PC
    ev = json.loads(runs["planted"]["evalset"].read_text())
    r = PC.Run(runs["out_planted"], ev)
    with np.load(runs["out_planted"] / "pairs_per_region.npz") as z:
        for k in ("log2_depth_ratio",) + S.PAIR_META_STR + S.PAIR_META_INT:
            assert k in z.files and len(z[k]) == len(r.pair_ids)
    full = PC.values(_load(runs["out_planted"]))
    got = PC.values(PC._agg(r, np.ones(len(r.region_indices)), S.pair_metrics_from_records,
                            ev["scorable"]))

    def close(a, b):
        return (a is None and b is None) or (a is not None and b is not None
                                             and a == pytest.approx(b, rel=1e-9, abs=1e-12))
    assert close(got["h4"], full["h4"]) and close(got["h5"], full["h5"])
    assert set(got["h4_assay"]) == set(full["h4_assay"])
    assert all(close(got["h4_assay"][a], full["h4_assay"][a]) for a in got["h4_assay"])
    assert got["rep"] and all(close(v, full["rep"][g]) for g, v in got["rep"].items())
