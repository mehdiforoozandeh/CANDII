"""Tests for tools/onion_eval/pareto_select.py (decision D1, D21) on synthetic arm summaries."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import pareto_select as P  # noqa: E402

TERMS = P.TERMS
REFS = P.DEFAULT_REFS

# fact 1 of the t370 plan: eic_v1_2, 256 regions; mean (seed range) per term
FACT1 = {
    "convae": ((0.3704, 0.0214), (0.6636, 0.0312), (0.5986, 0.1864)),
    "convattn": ((0.3860, 0.0298), (0.6750, 0.0193), (0.6272, 0.1379)),
    "unet": ((0.4452, 0.0380), (0.5352, 0.0624), (0.6460, 0.1577)),
}


def row(arm, terms, n=3):
    r = {"arm": arm, "n_seeds": str(n)}
    for k, (m, g) in zip(TERMS, terms):
        r[f"{k}.mean"] = "" if m is None else f"{m:.4f}"
        r[f"{k}.range"] = "" if g is None else f"{g:.4f}"
    return r


def refs_rows():
    return [row(REFS[m], FACT1[m]) for m in P.MODELS]


def arm_of(res, name):
    return next(a for a in res["arms"] if a["arm"] == name)


def test_fact1_floors():
    f = P.select(refs_rows(), REFS)["floors"]
    assert f["imputation_raw"]["floor"] == pytest.approx(0.4072, abs=1e-9)
    assert f["correction_v3_raw"]["floor"] == pytest.approx(0.6557, abs=1e-9)
    assert f["uncertainty_raw"]["floor"] == pytest.approx(0.4883, abs=1e-9)
    assert [f[k]["best_model"] for k in TERMS] == ["unet", "convattn", "unet"]


def test_bases_themselves_do_not_win():
    # each base as an arm: U-Net fails correction, the convs fail imputation
    rows = refs_rows() + [row(f"rcp_copy_{m}", FACT1[m]) for m in P.MODELS]
    res = P.select(rows, REFS)
    assert res["winner_overall"] is None
    assert all(not a["wins"] for a in res["arms"])
    assert arm_of(res, "rcp_copy_unet")["class"] == "FAIL"   # 0.1205 short, range 0.0624
    assert arm_of(res, "rcp_copy_convattn")["class"] == "NEAR"  # 0.0212 short < range 0.0298


def test_win_and_overall_by_score():
    rows = refs_rows() + [
        # imputation beyond: 0.50 - 0.4452 = 0.0548 > max(0.01, 0.038)
        row("rcp_a_unet", ((0.50, 0.01), (0.68, 0.01), (0.65, 0.10))),
        # wins with a smaller min z
        row("rcp_b_convattn", ((0.49, 0.01), (0.66, 0.01), (0.65, 0.10))),
        # eligible, nothing beyond
        row("rcp_c_convae", ((0.45, 0.01), (0.67, 0.01), (0.60, 0.10))),
    ]
    res = P.select(rows, REFS)
    a, b, c = (arm_of(res, n) for n in ("rcp_a_unet", "rcp_b_convattn", "rcp_c_convae"))
    assert a["wins"] and a["class"] == "PASS" and a["beyond_terms"] == "imputation_raw"
    assert b["wins"] and b["score"] < a["score"]
    assert c["eligible"] and not c["wins"] and c["class"] == "NEAR"
    assert res["winner_by_model"] == {"convae": None, "convattn": "rcp_b_convattn",
                                      "unet": "rcp_a_unet"}
    assert res["winner_overall"] == "rcp_a_unet"
    # z = (arm - best) / best range; score = min z
    zc = (0.66 - 0.6750) / 0.0193
    assert b["score"] == pytest.approx(min(b[f"{k}.z"] for k in TERMS))
    assert b["correction_v3_raw.z"] == pytest.approx(zc, abs=1e-6)


def test_beyond_uses_larger_range():
    # 0.5 - 0.4452 = 0.0548 < arm range 0.06: not beyond, so not a win
    rows = refs_rows() + [row("rcp_x_unet", ((0.50, 0.06), (0.68, 0.01), (0.65, 0.10)))]
    a = arm_of(P.select(rows, REFS), "rcp_x_unet")
    assert a["eligible"] and a["beyond_terms"] == "" and a["class"] == "NEAR"


def test_floor_exact_is_eligible():
    rows = refs_rows() + [row("rcp_e_unet", ((0.4072, 0.01), (0.6557, 0.01), (0.85, 0.01)))]
    a = arm_of(P.select(rows, REFS), "rcp_e_unet")
    assert a["eligible"] and a["wins"] and a["beyond_terms"] == "uncertainty_raw"


def test_near_one_term_below_by_less_than_own_range():
    rows = refs_rows() + [
        row("rcp_n_unet", ((0.40, 0.01), (0.70, 0.01), (0.70, 0.10))),   # 0.0072 < 0.01
        row("rcp_f_unet", ((0.40, 0.005), (0.70, 0.01), (0.70, 0.10))),  # 0.0072 > 0.005
        row("rcp_g_unet", ((0.40, 0.05), (0.60, 0.05), (0.70, 0.10))),   # two below
    ]
    res = P.select(rows, REFS)
    assert arm_of(res, "rcp_n_unet")["class"] == "NEAR"
    assert arm_of(res, "rcp_f_unet")["class"] == "FAIL"
    assert arm_of(res, "rcp_g_unet")["class"] == "FAIL"
    assert res["winner_overall"] is None


def test_wait_never_wins():
    rows = refs_rows() + [row("rcp_w_unet", ((0.6, 0.01), (0.8, 0.01), (0.8, 0.01)), n=2)]
    res = P.select(rows, REFS)
    assert arm_of(res, "rcp_w_unet")["class"] == "WAIT"
    assert res["winner_overall"] is None


def test_undefined_arm_term_fails():
    rows = refs_rows() + [row("rcp_u_unet", ((0.6, 0.01), (None, None), (0.8, 0.01)))]
    a = arm_of(P.select(rows, REFS), "rcp_u_unet")
    assert not a["eligible"] and a["class"] == "FAIL" and a["score"] is None


def test_own_base_info():
    rows = refs_rows() + [row("rcp_o_unet", ((0.50, 0.01), (0.68, 0.01), (0.65, 0.10)))]
    a = arm_of(P.select(rows, REFS), "rcp_o_unet")
    assert a["own_diff.correction_v3_raw"] == pytest.approx(0.68 - 0.5352, abs=1e-6)
    assert a["own_beyond.correction_v3_raw"] is True      # 0.1448 > 0.0624
    assert a["own_beyond.uncertainty_raw"] is False


def test_depth_native_row_drives_eligibility():
    main = row("rcp_d_unet", ((0.50, 0.01), (0.60, 0.01), (0.65, 0.10)))   # correction below
    native = [row("rcp_d_unet", ((0.50, 0.01), (0.70, 0.01), (0.65, 0.10)))]
    rows = refs_rows() + [main]
    assert not arm_of(P.select(rows, REFS), "rcp_d_unet")["eligible"]
    res = P.select(rows, REFS, depth_native=native)
    a = arm_of(res, "rcp_d_unet")
    assert a["eligible"] and a["wins"] and a["eligibility_row"] == "depth_native"
    assert a["correction_v3_raw.mean"] == pytest.approx(0.60)   # reported from the main row


@pytest.mark.parametrize("bad", ["missing", "seeds", "undefined"])
def test_reference_errors(bad):
    rows = refs_rows()
    if bad == "missing":
        rows = rows[:2]
    elif bad == "seeds":
        rows[0]["n_seeds"] = "2"
    else:
        rows[1]["uncertainty_raw.range"] = ""
    with pytest.raises(P.RefError):
        P.select(rows, REFS)


def _write(path, rows):
    cols = ["arm", "n_seeds", "composite_version"] + [
        f"{k}.{s}" for k in TERMS for s in ("mean", "range", "n_defined")]
    lines = ["\t".join(cols)] + ["\t".join(r.get(c, "") for c in cols) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_cli_exit_codes(tmp_path, capsys):
    s = tmp_path / "s.tsv"
    win = row("rcp_a_unet", ((0.50, 0.01), (0.68, 0.01), (0.65, 0.10)))
    other = row("zzz_a_unet", ((0.50, 0.01), (0.68, 0.01), (0.65, 0.10)))
    _write(s, refs_rows() + [win, other])
    out = tmp_path / "o.tsv"
    assert P.main(["--summary", str(s), "--arm-prefix", "rcp_", "--out", str(out)]) == 0
    text = out.read_text().splitlines()
    assert text[0].split("\t")[:3] == ["arm", "model", "n_seeds"]
    assert len(text) == 2 and text[1].startswith("rcp_a_unet\tunet\t3")   # prefix filter
    assert "winner overall: rcp_a_unet" in capsys.readouterr().err

    s1 = tmp_path / "s1.tsv"
    _write(s1, refs_rows() + [row("rcp_copy_unet", FACT1["unet"])])
    assert P.main(["--summary", str(s1)]) == 1

    s3 = tmp_path / "s3.tsv"
    _write(s3, refs_rows()[:2] + [win])
    assert P.main(["--summary", str(s3)]) == 3


def test_cli_multiple_summaries_and_depth_native(tmp_path):
    a, b, n = tmp_path / "a.tsv", tmp_path / "b.tsv", tmp_path / "n.tsv"
    _write(a, refs_rows())
    _write(b, [row("rcp_d_unet", ((0.50, 0.01), (0.60, 0.01), (0.65, 0.10)))])
    _write(n, [row("rcp_d_unet", ((0.50, 0.01), (0.70, 0.01), (0.65, 0.10)))])
    assert P.main(["--summary", str(a), "--summary", str(b)]) == 1
    assert P.main(["--summary", str(a), "--summary", str(b),
                   "--depth-native-summary", str(n)]) == 0
