"""t396 C8 and iteration 2 A4: tools/t396/score_spaces.py (records, table, check) on synthetic
arrays in the contract format of tools/t396/dump_native.py.

Requirements checked:
  - a model whose mean is the true rate gets skill near 1 in all three spaces;
  - `--regions pair` and `even` / `odd` pick the right regions;
  - the count-output path and the start-output path agree in sum_1kb_count when one is the exact
    convolution of the other;
  - a count-output arm writes no start_sum row;
  - `check` passes on the right scores.json and exits 1 on a wrong one (gate b), native rows only;
  - `table` columns (space, model first; calibration; every_seed_above), the macros
    (H3K36me3 equals that assay's row), the read-length split and the truth validation (gate c);
  - calibration: with identity rho and c = 1 the native and calibrated rows agree; the out-of-fold
    rule never uses a region's own half, for rho and for c; a count-output arm gets a calibrated
    row.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import deviance as DV  # noqa: E402
from t396 import calib_cov as CV  # noqa: E402
from t396 import covconv as CC  # noqa: E402
from t396 import deconv as DC  # noqa: E402
from t396 import score_spaces as SS  # noqa: E402

R, RB = 6, 400
ASSAYS = ["H3K4me3", "H3K27ac", "ATAC-seq"]
SCORABLE = ["H3K4me3", "H3K27ac"]
PAIR = [1, 4]


def _kernel():
    w = np.zeros(17)
    w[8:13] = [0.5, 1.0, 1.0, 0.8, 0.3]      # reads cover the start bin and 4 bins downstream
    return np.stack([w, w])


def _rate(rng):
    """Per-strand start rate [R, RB, 2]: blocks of high and low signal, region-specific."""
    blocks = rng.gamma(0.6, 6.0, size=(R, RB // 20, 1))
    lam = np.repeat(blocks, 20, axis=1) + 0.05
    return np.concatenate([lam * 0.5, lam * 0.5], axis=-1)


def _conv_strands(lam, k):
    return DC.forward(lam[..., 0], k[0]) + DC.forward(lam[..., 1], k[1])


def _key_file(key):
    return hashlib.sha1(key.encode()).hexdigest() + ".npz"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("w")
    rng = np.random.default_rng(0)
    k = _kernel()
    truth, arrays, store = root / "truth", root / "arrays", root / "store"
    for d in (truth, arrays, store):
        d.mkdir()
    rates = {}
    for ti in range(6):
        assay = ASSAYS[ti % 3]
        tid = f"T{ti}"
        lam = _rate(rng)
        cr = _conv_strands(lam, k)                         # true overlap rate
        f1, f2 = 0.45, 0.55
        s1 = rng.poisson(f1 * lam)
        s2 = rng.poisson(f2 * lam)
        s = s1 + s2
        y1 = np.rint(_conv_strands(s1.astype(float), k)).astype(np.int32)
        y2 = np.rint(_conv_strands(s2.astype(float), k)).astype(np.int32)
        y = y1 + y2
        prof = 1.0 + 0.5 * np.sin(np.arange(RB) / 15.0)    # a smooth position average
        pa_s = np.broadcast_to(prof * lam.sum(-1).mean(), (R, RB))
        np.savez(truth / f"{tid}.npz", mid=np.array(tid), assay=np.array(assay),
                 cell=np.array("c"), stranded=np.array(assay != "ATAC-seq"),
                 read_length=np.array(36 if ti < 4 else 76),
                 chrom=np.array(["chr1"] * R), start_bin=np.arange(R) * 10000,
                 depth_reads=np.array(1e6), shares=np.array([f1, f2]),
                 shares_starts=np.array([f1, f2]), kernel=k, n_reads_strand=np.array([5e5, 5e5]),
                 y=y, y_pr1=y1, y_pr2=y2, s=s.astype(np.uint16), s_pr1=s1.astype(np.uint16),
                 s_pr2=s2.astype(np.uint16), pa_counts=DC.forward(pa_s, k[0]).astype(np.float32),
                 pa_starts=pa_s.astype(np.float32), pa_counts_ids=np.array(["a"]),
                 pa_starts_ids=np.array(["a"]))
        mu_sub = lam.copy()
        if assay == "ATAC-seq":                            # unstranded: total in channel 0
            mu_sub = np.stack([lam.sum(-1), np.zeros((R, RB))], -1)
        n_big = np.full((R, RB), 1e4)
        np.savez(arrays / f"{tid}.npz", mu_sub=mu_sub.astype(np.float32),
                 n_sub=np.full((R, RB, 2), 1e4, np.float32), mu=cr.astype(np.float32),
                 n=n_big.astype(np.float32), max_abs_rel_diff=np.array(0.0),
                 region_indices=np.arange(R))
        key = f"item/{tid}/leave_one_assay_out"
        np.savez(store / _key_file(key), key=np.array(key), mu=cr.astype(np.float32),
                 n=n_big.astype(np.float32))
        rates[tid] = (lam, cr, assay)
    ev = root / "evalset.json"
    ev.write_text(json.dumps({"pair_regions": PAIR}))
    return {"root": root, "truth": truth, "arrays": arrays, "store": store, "evalset": ev,
            "rates": rates}


def _expected_skill(world, scorable=SCORABLE):
    """dev_imputation_skill macro straight from deviance.target_terms (not via spaces.aggregate)."""
    per = {}
    for p in sorted(world["truth"].glob("*.npz")):
        z = np.load(p)
        _, cr, assay = world["rates"][p.stem]
        t = DV.target_terms(z["y"].ravel().astype(float), cr.astype(np.float32).ravel(),
                            np.full(cr.size, 1e4), mu_ref=z["pa_counts"].ravel())
        per.setdefault(assay, []).append(t)
    return float(np.mean([DV.gap(per[a], "D_ref") for a in scorable]))


def _scores(world, name, skill):
    p = world["root"] / f"{name}.json"
    p.write_text(json.dumps({"deviance": {"dev_imputation_skill": {"macro": skill}},
                             "scorable": {"imputation": SCORABLE}, "n_targets": 6,
                             "evalset": str(world["evalset"])}))
    return p


def _records(world, which, seed, regions=("all", "pair", "even", "odd"), validate=False,
             skill=None, tag=""):
    sj = _scores(world, f"scores{tag}", _expected_skill(world) if skill is None else skill)
    out = world["root"] / f"rec_{which}_{seed}{tag}.jsonl"
    args = ["records", "--truth", str(world["truth"]), "--model", which, "--seed", str(seed),
            "--scores-json", str(sj), "--regions", *regions, "--out", str(out)]
    args += (["--arrays", str(world["arrays"])] if which == "rs"
             else ["--pred-store", str(world["store"])])
    if validate:
        args.append("--validate-truth")
    assert SS.main(args) == 0
    return out, [json.loads(x) for x in out.read_text().splitlines()]


@pytest.fixture(scope="module")
def recs(world):
    rs, rs_rows = _records(world, "rs", 0)
    ref, ref_rows = _records(world, "ref", 0, validate=True)
    return {"rs": (rs, rs_rows), "ref": (ref, ref_rows)}


def _macro_skill(rows, space, regions, n_iter=None, model=None):
    from t396 import spaces as SP
    sel = [r for r in rows if r["kind"] == "score" and r["space"] == space
           and r["regions"] == regions and r["n_iter"] == n_iter]
    return SP.aggregate(sel, SCORABLE)["scorable"]["skill"]


def test_true_rate_skill_near_one_all_spaces(recs):
    rows = recs["rs"][1]
    for space in ("overlap_count", "start_sum", "sum_1kb_count"):
        assert _macro_skill(rows, space, "all") > 0.9, space


def test_count_output_arm_has_no_start_sum_row(recs):
    score = [r for r in recs["ref"][1] if r["kind"] == "score"]
    assert score and not any(r["space"] == "start_sum" for r in score)
    assert {r["path"] for r in score} == {"count_output"}
    assert {r["path"] for r in recs["rs"][1]} == {"start_output"}
    assert any(r["space"] == "start_sum" for r in recs["rs"][1])
    for r in score + recs["rs"][1]:
        assert r["calibration"] == "native" and r["n_iter"] is None
    for r in recs["ref"][1]:
        assert {"assay", "read_length", "regions"} <= set(r)


def test_region_selection(world, recs):
    from t396 import spaces as SP
    rows = recs["rs"][1]
    z = np.load(world["truth"] / "T0.npz")
    _, cr, _ = world["rates"]["T0"]
    for name, idx in (("pair", PAIR), ("even", [0, 2, 4]), ("odd", [1, 3, 5]), ("all", range(R))):
        idx = list(idx)
        r = [x for x in rows if x["target"] == "T0" and x["space"] == "overlap_count"
             and x["regions"] == name][0]
        want = SP.score_target(z["y"][idx], cr.astype(np.float32)[idx], np.full((len(idx), RB), 1e4),
                               z["pa_counts"][idx], z["y_pr1"][idx], z["y_pr2"][idx], z["shares"])
        assert r["bins"] == len(idx) * RB
        assert r["D_model"] == pytest.approx(want["D_model"], rel=1e-9)
        assert r["D_ref"] == pytest.approx(want["D_ref"], rel=1e-9)


def test_reference_and_read_start_agree_in_1kb(recs):
    a = {r["target"]: r for r in recs["rs"][1]
         if r["space"] == "sum_1kb_count" and r["regions"] == "all"}
    b = {r["target"]: r for r in recs["ref"][1]
         if r["kind"] == "score" and r["space"] == "sum_1kb_count" and r["regions"] == "all"}
    assert set(a) == set(b) and a
    for t in a:
        for k in ("D_model", "D_ref", "E_nb", "D_model_pr2"):
            assert a[t][k] == pytest.approx(b[t][k], rel=1e-12)


def test_check_gate_b(world, recs, tmp_path):
    out = tmp_path / "check.tsv"
    assert SS.main(["check", "--records", str(recs["rs"][0]), str(recs["ref"][0]),
                    "--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 3 and all(x.split("\t")[-1] == "pass" for x in lines[1:])
    bad, _ = _records(world, "rs", 1, regions=("all",), skill=_expected_skill(world) + 0.01,
                      tag="_bad")
    assert SS.main(["check", "--records", str(bad), "--out", str(tmp_path / "c2.tsv")]) == 1
    assert (tmp_path / "c2.tsv").read_text().splitlines()[1].split("\t")[-1] == "fail"


def test_shards_cover_targets(world, tmp_path):
    sj = _scores(world, "scores_sh", 0.0)
    got = []
    for i in range(2):
        out = tmp_path / f"s{i}.jsonl"
        assert SS.main(["records", "--truth", str(world["truth"]), "--model", "rs", "--seed", "0",
                        "--arrays", str(world["arrays"]), "--scores-json", str(sj),
                        "--regions", "all", "--out", str(out), "--shard", f"{i}/2"]) == 0
        got.append({json.loads(x)["target"] for x in out.read_text().splitlines()})
    assert not (got[0] & got[1]) and len(got[0] | got[1]) == 6


def _read_tsv(p):
    lines = Path(p).read_text().splitlines()
    cols = lines[0].split("\t")
    return cols, [dict(zip(cols, x.split("\t"))) for x in lines[1:]]


def test_table_columns_and_flags(recs, tmp_path):
    out = tmp_path / "t"
    assert SS.main(["table", "--records", str(recs["rs"][0]), str(recs["ref"][0]),
                    "--out", str(out)]) == 0
    cols, rows = _read_tsv(out / "table.tsv")
    assert cols[:3] == ["space", "model", "calibration"]
    for c in ("regions", "n_iter", "skill", "r2", "calibration_ratio", "ratio_to_halves", "floor",
              "seed_range", "per_seed", "macro", "every_seed_above", "flag"):
        assert c in cols
    assert cols[-1] == "flag"
    assert not [r for r in rows if r["model"] == "ref" and r["space"] == "start_sum"]
    assert {r["macro"] for r in rows} == {"scorable", "all_assays", "scorable_without_H3K36me3"}
    assert all(r["flag"] == "" and r["calibration"] == "native" for r in rows)
    _, tv = _read_tsv(out / "truth_validation.tsv")
    allrow = [r for r in tv if r["assay"] == "ALL" and r["n_iter"] == "50"]
    assert allrow and allrow[0]["gate_c"] == "pass"
    _, rl = _read_tsv(out / "per_read_length.tsv")
    assert {r["read_length_group"] for r in rl} == {"36bp", "longer"}
    _, pa = _read_tsv(out / "per_assay.tsv")
    assert {r["assay"] for r in pa} == set(ASSAYS)


def test_table_gate_c_fail_in_truth_validation(recs, tmp_path):
    rows = recs["ref"][1]
    bad = tmp_path / "bad.jsonl"
    with open(bad, "w") as fh:
        for r in rows:
            if r["kind"] == "validate" and r["assay"] == "H3K4me3":
                r = dict(r, roundtrip_nmse=0.2)
            fh.write(json.dumps(r) + "\n")
    out = tmp_path / "t"
    assert SS.main(["table", "--records", str(bad), "--out", str(out)]) == 0
    _, tv = _read_tsv(out / "truth_validation.tsv")
    assert [r["gate_c"] for r in tv if r["assay"] == "H3K4me3" and r["n_iter"] == "50"] == ["fail"]
    out2 = tmp_path / "t2"
    assert SS.main(["table", "--records", str(recs["rs"][0]), "--out", str(out2)]) == 0


def test_h3k36me3_macro_equals_its_assay_row(recs, tmp_path):
    # relabel ATAC-seq as H3K36me3 and make it scorable
    p = tmp_path / "k36.jsonl"
    with open(p, "w") as fh:
        for r in recs["rs"][1]:
            r = dict(r, scorable=SCORABLE + ["H3K36me3"])
            if r["assay"] == "ATAC-seq":
                r["assay"] = "H3K36me3"
            fh.write(json.dumps(r) + "\n")
    out = tmp_path / "t"
    assert SS.main(["table", "--records", str(p), "--out", str(out)]) == 0
    _, rows = _read_tsv(out / "table.tsv")
    _, pa = _read_tsv(out / "per_assay.tsv")
    for space in ("overlap_count", "start_sum", "sum_1kb_count"):
        sel = {r["macro"]: r for r in rows if r["space"] == space and r["regions"] == "all"}
        a = [r for r in pa if r["space"] == space and r["regions"] == "all"]
        k36 = next(r for r in a if r["assay"] == "H3K36me3")
        assert sel["H3K36me3"]["skill"] == k36["skill"] and sel["H3K36me3"]["n_assays"] == "1"
        others = [float(r["skill"]) for r in a if r["assay"] in SCORABLE]
        assert float(sel["scorable_without_H3K36me3"]["skill"]) == pytest.approx(np.mean(others))
        assert sel["scorable"]["n_assays"] == "3"
        assert sel["scorable_without_H3K36me3"]["n_assays"] == "2"


def test_every_seed_above_constructed():
    base = {"space": "start_sum", "calibration": "native", "regions": "all", "n_iter": None,
            "macro": "scorable"}
    trows = [dict(base, model=m) for m in ("a", "b", "c", "d")]
    trows.append(dict(base, model="a", space="overlap_count"))
    seeds = [[0.50, 0.52, 0.51],          # a: above b and c
             [0.40, 0.45, 0.49],          # b: above c only
             [0.30, 0.39, 0.35],          # c: above none
             [0.49, 0.60, 0.70],          # d: overlaps b (0.49 = max of b, not strictly below)
             [0.10, 0.10, 0.10]]          # another space: never compared
    SS.every_seed_above(trows, seeds)
    got = {(r["space"], r["model"]): r["every_seed_above"] for r in trows}
    assert got[("start_sum", "a")] == "b,c"
    assert got[("start_sum", "b")] == "c"
    assert got[("start_sum", "c")] == ""
    assert got[("start_sum", "d")] == "c"
    assert got[("overlap_count", "a")] == ""


def test_every_seed_above_in_table(world, recs, tmp_path):
    p1 = tmp_path / "worse.jsonl"
    with open(p1, "w") as fh:
        for seed in (0, 1, 2):
            for r in recs["rs"][1]:
                fh.write(json.dumps(dict(r, model="worse", seed=seed,
                                         D_model=r["D_model"] * (1.3 + 0.01 * seed))) + "\n")
    p2 = tmp_path / "good.jsonl"
    with open(p2, "w") as fh:
        for seed in (0, 1, 2):
            for r in recs["rs"][1]:
                fh.write(json.dumps(dict(r, model="good", seed=seed,
                                         D_model=r["D_model"] * (1.0 + 0.01 * seed))) + "\n")
    out = tmp_path / "t"
    assert SS.main(["table", "--records", str(p1), str(p2), "--out", str(out)]) == 0
    _, rows = _read_tsv(out / "table.tsv")
    sel = {r["model"]: r for r in rows if r["space"] == "overlap_count" and r["regions"] == "all"
           and r["macro"] == "scorable"}
    assert sel["good"]["every_seed_above"] == "worse" and sel["worse"]["every_seed_above"] == ""


def test_seed_range(world, recs, tmp_path):
    p1, rows1 = _records(world, "rs", 7, regions=("all",), tag="_s7")
    # perturb seed 7 so its skill differs
    with open(p1, "w") as fh:
        for r in rows1:
            fh.write(json.dumps(dict(r, D_model=r["D_model"] * 1.05)) + "\n")
    out = tmp_path / "t"
    assert SS.main(["table", "--records", str(recs["rs"][0]), str(p1), "--out", str(out)]) == 0
    _, rows = _read_tsv(out / "table.tsv")
    r = [x for x in rows if x["space"] == "overlap_count" and x["regions"] == "all"
         and x["macro"] == "scorable"][0]
    assert float(r["seed_range"]) > 0 and r["per_seed"].count(":") == 2


# ------------------------------------------------------------------ calibration (iteration 2 A4)
def _write_fit(path, half, world, *, c=1.0, num=None, den=None, seed=0, has_arrays=True,
               has_ref=True):
    """A fit file in the format of calib_cov.py fit, written by hand. num = den = 0 gives the
    identity rho (finish_rho)."""
    tids = sorted(world["rates"])
    T = len(tids)
    shp = (T, 2, 2, CC.NLAG)
    num = np.zeros(shp) if num is None else np.broadcast_to(num, shp)
    den = np.zeros(shp) if den is None else np.broadcast_to(den, shp)
    np.savez(path, seed=np.array(seed), targets=np.array(tids),
             assays_t=np.array([world["rates"][t][2] for t in tids]),
             stranded=np.array([world["rates"][t][2] != "ATAC-seq" for t in tids]),
             num=num, den=den, c_rs_loo=np.full(T, np.nan), c_ref_loo=np.full(T, float(c)),
             fallback=np.zeros(T, bool), window_bins=np.array(RB), infer_batch=np.array(1),
             fit_regions=np.array(half), has_arrays=np.array(has_arrays),
             has_ref=np.array(has_ref))
    return path


@pytest.fixture(scope="module")
def cworld(world, tmp_path_factory):
    """Arrays whose stored mu, n are the identity-rho conversion of mu_sub, n_sub (one segment
    per region: window_bins = RB, infer_batch = 1), and a count store holding the same mu, n."""
    root = tmp_path_factory.mktemp("cw")
    arrays, store = root / "arrays", root / "store"
    arrays.mkdir()
    store.mkdir()
    for tid in sorted(world["rates"]):
        tr = np.load(world["truth"] / f"{tid}.npz")
        z = np.load(world["arrays"] / f"{tid}.npz")
        stranded = bool(tr["stranded"])
        w = CV.kernel_row(tr["kernel"], tr["n_reads_strand"], stranded)
        mu_sub = z["mu_sub"].astype(np.float64)
        n_sub = np.full(mu_sub.shape, 2.0)
        mu = np.empty((R, RB))
        n = np.empty((R, RB))
        for r in range(R):
            mu[r], n[r] = CC.convert_nb_cov(mu_sub[r], n_sub[r], w, stranded, CC.identity_rho())
        np.savez(arrays / f"{tid}.npz", mu_sub=mu_sub.astype(np.float32),
                 n_sub=n_sub.astype(np.float32), mu=mu.astype(np.float32),
                 n=n.astype(np.float32), region_indices=np.arange(R))
        key = f"item/{tid}/leave_one_assay_out"
        np.savez(store / _key_file(key), key=np.array(key), mu=mu.astype(np.float32),
                 n=n.astype(np.float32))
    return {"root": root, "arrays": arrays, "store": store}


def _crecords(world, cw, which, odd, even, name, regions=("all", "even", "odd")):
    sj = _scores(world, "scores_cal", 0.0)
    out = cw["root"] / f"{name}.jsonl"
    args = ["records", "--truth", str(world["truth"]), "--model", which, "--seed", "0",
            "--scores-json", str(sj), "--regions", *regions, "--out", str(out),
            "--rho-odd", str(odd), "--rho-even", str(even)]
    args += (["--arrays", str(cw["arrays"])] if which == "rs" else ["--pred-store", str(cw["store"])])
    assert SS.main(args) == 0
    return [json.loads(x) for x in out.read_text().splitlines()]


def test_identity_rho_and_unit_c_rows_agree(world, cworld):
    odd = _write_fit(cworld["root"] / "id_odd.npz", "odd", world)
    even = _write_fit(cworld["root"] / "id_even.npz", "even", world)
    for which in ("rs", "ref"):
        rows = _crecords(world, cworld, which, odd, even, f"id_{which}")
        nat = {(r["target"], r["space"], r["regions"]): r for r in rows
               if r["calibration"] == "native"}
        cal = {(r["target"], r["space"], r["regions"]): r for r in rows
               if r["calibration"] == "calibrated"}
        assert set(cal) == {k for k in nat if k[1] != "start_sum"} and cal
        for k, r in cal.items():
            for t in ("D_model", "D_ref", "E_nb", "D_model_pr2"):
                assert r[t] == pytest.approx(nat[k][t], rel=1e-5), (which, k, t)


def test_count_output_gets_calibrated_row_out_of_fold_c(world, cworld):
    from t396 import spaces as SP
    odd = _write_fit(cworld["root"] / "c_odd.npz", "odd", world, c=2.0, has_arrays=False)
    even = _write_fit(cworld["root"] / "c_even.npz", "even", world, c=5.0, has_arrays=False)
    rows = _crecords(world, cworld, "ref", odd, even, "c_ref")
    cal = [r for r in rows if r["calibration"] == "calibrated"]
    assert {r["space"] for r in cal} == {"overlap_count", "sum_1kb_count"}
    assert all(r["c_odd_fit"] == 2.0 and r["c_even_fit"] == 5.0 for r in cal)
    # direct: an even region takes the odd fit's c, an odd region the even fit's
    st = SS.read_store(cworld["store"], "T0", (R, RB))
    fits = SS.Fits(odd, even, 0, SS.PATH_COUNT)
    tr = SS.load_npz(world["truth"] / "T0.npz")
    n_cal, fb, _ = SS.calibrated_n(tr, "T0", list(range(R)), fits, n_c=st["n"])
    assert not fb
    assert np.allclose(n_cal[0::2], st["n"][0::2] / 2.0)
    assert np.allclose(n_cal[1::2], st["n"][1::2] / 5.0)
    # through records: the even-region row equals score_target with n / 2
    z = np.load(world["truth"] / "T0.npz")
    ev = [0, 2, 4]
    want = SP.score_target(z["y"][ev], st["mu"][ev], st["n"][ev] / 2.0, z["pa_counts"][ev],
                           z["y_pr1"][ev], z["y_pr2"][ev], z["shares"])
    got = next(r for r in cal if r["target"] == "T0" and r["space"] == "overlap_count"
               and r["regions"] == "even")
    assert got["E_nb"] == pytest.approx(want["E_nb"], rel=1e-9)


def test_out_of_fold_rho_never_uses_own_half(world, cworld):
    shp = (2, 2, CC.NLAG)
    odd = _write_fit(cworld["root"] / "r_odd.npz", "odd", world)              # identity
    even = _write_fit(cworld["root"] / "r_even.npz", "even", world,           # rho 0.5 off lag 0
                      num=np.full(shp, 0.5), den=np.ones(shp))
    fits = SS.Fits(odd, even, 0, SS.PATH_START)
    tr = SS.load_npz(world["truth"] / "T0.npz")
    m = np.load(cworld["arrays"] / "T0.npz")
    n_st = m["n"].astype(np.float64)
    n_cal, fb, prm = SS.calibrated_n(tr, "T0", list(range(R)), fits, n_c=n_st,
                                     mu_sub=m["mu_sub"].astype(np.float64),
                                     n_sub=m["n_sub"].astype(np.float64))
    assert np.allclose(n_cal[0::2], n_st[0::2], rtol=1e-5)   # even regions: the odd (identity) fit
    assert not np.allclose(n_cal[1::2], n_st[1::2], rtol=1e-2)
    assert (n_cal[1::2] < n_st[1::2]).mean() > 0.9           # positive rho: more variance
    for half in ("odd", "even"):                              # leave one target out
        assert "T0" not in prm[f"rho_from_{half}_fit"] and prm[f"rho_from_{half}_fit"]
    # changing the even fit's parts leaves the even regions unchanged
    even2 = _write_fit(cworld["root"] / "r_even2.npz", "even", world,
                       num=np.full(shp, -0.3), den=np.ones(shp))
    n2, _, _ = SS.calibrated_n(tr, "T0", list(range(R)), SS.Fits(odd, even2, 0, SS.PATH_START),
                               n_c=n_st, mu_sub=m["mu_sub"].astype(np.float64),
                               n_sub=m["n_sub"].astype(np.float64))
    assert np.array_equal(n2[0::2], n_cal[0::2]) and not np.allclose(n2[1::2], n_cal[1::2])


def test_fit_halves_checked(world, cworld):
    odd = _write_fit(cworld["root"] / "h_odd.npz", "odd", world)
    with pytest.raises(SystemExit):
        SS.Fits(odd, odd, 0, SS.PATH_START)               # an odd fit given as --rho-even
    nor = _write_fit(cworld["root"] / "h_even.npz", "even", world, has_arrays=False)
    with pytest.raises(SystemExit):
        SS.Fits(odd, nor, 0, SS.PATH_START)               # no rho parts for a start-output arm
    with pytest.raises(SystemExit):
        SS.Fits(odd, _write_fit(cworld["root"] / "h_s1.npz", "even", world, seed=1), 0,
                SS.PATH_COUNT)


def test_check_ignores_calibrated_rows(world, recs, tmp_path):
    odd = _write_fit(tmp_path / "o.npz", "odd", world, c=3.0, has_arrays=False)
    even = _write_fit(tmp_path / "e.npz", "even", world, c=3.0, has_arrays=False)
    sj = _scores(world, "scores_chk", _expected_skill(world))
    out = tmp_path / "rec.jsonl"
    assert SS.main(["records", "--truth", str(world["truth"]), "--model", "ref", "--seed", "0",
                    "--pred-store", str(world["store"]), "--scores-json", str(sj),
                    "--regions", "all", "--out", str(out), "--rho-odd", str(odd),
                    "--rho-even", str(even)]) == 0
    assert SS.main(["check", "--records", str(out), "--out", str(tmp_path / "c.tsv")]) == 0
    t = tmp_path / "t"
    assert SS.main(["table", "--records", str(out), "--out", str(t)]) == 0
    _, rows = _read_tsv(t / "table.tsv")
    assert {r["calibration"] for r in rows} == {"native", "calibrated"}
