"""t396 C8: tools/t396/score_spaces.py (records, table, check) on synthetic arrays in the contract
format of tools/t396/dump_native.py.

Requirements checked:
  - a model whose mean is the true rate gets skill near 1 in all three spaces;
  - `--regions pair` and `even` / `odd` pick the right regions;
  - the reference path and the read-start path agree in sum_1kb_count when one is the exact
    convolution of the other;
  - `check` passes on the right scores.json and exits 1 on a wrong one (gate b);
  - `table` columns and flags (conversion_artifact, primary, best_over_grid, not_interpreted from
    gate c), the read-length split and the truth validation.
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


def test_reference_start_space_rows_per_n_iter(recs):
    rows = [r for r in recs["ref"][1] if r["kind"] == "score" and r["space"] == "start_sum"
            and r["regions"] == "all"]
    assert sorted({r["n_iter"] for r in rows}) == list(DC.N_ITER_GRID)
    assert _macro_skill(recs["ref"][1], "start_sum", "all", 200) > 0.5
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
    for c in ("space", "model", "regions", "n_iter", "skill", "r2", "calibration_ratio",
              "ratio_to_halves", "floor", "seed_range", "per_seed", "macro", "flag"):
        assert c in cols
    assert cols[-1] == "flag"
    ref_start = [r for r in rows if r["model"] == "ref" and r["space"] == "start_sum"]
    assert ref_start and all("conversion_artifact" in r["flag"] for r in ref_start)
    for mac in ("scorable", "all_assays"):
        sel = [r for r in ref_start if r["regions"] == "all" and r["macro"] == mac]
        assert sum("best_over_grid" in r["flag"] for r in sel) == 1
        assert [r["n_iter"] for r in sel if "primary" in r["flag"]] == ["50"]
    others = [r for r in rows if not (r["model"] == "ref" and r["space"] == "start_sum")]
    assert all(r["flag"] == "" for r in others)
    # gate (c): the clean synthetic truth passes, so nothing is not_interpreted
    _, tv = _read_tsv(out / "truth_validation.tsv")
    allrow = [r for r in tv if r["assay"] == "ALL" and r["n_iter"] == "50"]
    assert allrow and allrow[0]["gate_c"] == "pass"
    assert not any("not_interpreted" in r["flag"] for r in rows)
    _, rl = _read_tsv(out / "per_read_length.tsv")
    assert {r["read_length_group"] for r in rl} == {"36bp", "longer"}
    _, pa = _read_tsv(out / "per_assay.tsv")
    assert {r["assay"] for r in pa} == set(ASSAYS)


def test_table_gate_c_fail_flags_not_interpreted(recs, tmp_path):
    rows = recs["ref"][1]
    bad = tmp_path / "bad.jsonl"
    with open(bad, "w") as fh:
        for r in rows:
            if r["kind"] == "validate" and r["assay"] == "H3K4me3":
                r = dict(r, roundtrip_nmse=0.2)
            fh.write(json.dumps(r) + "\n")
    out = tmp_path / "t"
    assert SS.main(["table", "--records", str(bad), "--out", str(out)]) == 0
    _, rows_t = _read_tsv(out / "table.tsv")
    ref_start = [r for r in rows_t if r["space"] == "start_sum"]
    assert ref_start and all("not_interpreted" in r["flag"] for r in ref_start)
    assert not any("not_interpreted" in r["flag"] for r in rows_t if r["space"] != "start_sum")
    _, tv = _read_tsv(out / "truth_validation.tsv")
    assert [r["gate_c"] for r in tv if r["assay"] == "H3K4me3" and r["n_iter"] == "50"] == ["fail"]
    # no validation rows at all: gate not evaluated, start rows still flagged
    out2 = tmp_path / "t2"
    assert SS.main(["table", "--records", str(recs["rs"][0]), "--out", str(out2)]) == 0


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
