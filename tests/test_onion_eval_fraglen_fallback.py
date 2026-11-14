"""FL: the estimated fragment length of a single-end ChIP-seq track whose `frag_len` is empty.

Pins:
  - `with_fraglen`: a filled `frag_len` is never replaced (the same row object comes back); an
    empty one takes a plausible estimate of the side table; an implausible estimate (outside
    80-400 bp, or xcor quality tag -2) is not used;
  - the scorer: with the side table, a target whose `frag_len` was blanked and is restored by an
    estimate EQUAL to the blanked value scores exactly as the original tables (a second route);
    targets with a table value are untouched by the option; without the option the blanked
    targets stay undefined ("covariates lack ['frag_len']") and no new key enters scores.json;
    `fraglen_source` per target and its counts;
  - the dump writes the truth of a target whose prediction has no signal (`truth_only`);
  - `ag2_baselines.run`: a training track with no fragment length is left out and counted, not
    fatal; with the side table it is used, and the training mean equals the original tables'."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import evalset as E  # noqa: E402
from onion_eval import signal_from_counts as SFC  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "DNase-seq"), (7, "chipseq-control"))
KW = dict(n_boot=10, guardrail_windows=4, log=lambda *a: None)
RB = 256
SIG = "signal_arcsinh"
H1 = "specific_variance_recovered_above_noise"
FL_COLS = ["measurement_id", "frag_len_estimated", "quality_tag"]


def _tsv(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _write_tsv(p, rows, cols=None):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols or list(rows[0]), delimiter="\t",
                           lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------------- unit

def test_with_fraglen_rule():
    tab = {"a": {"measurement_id": "a", "frag_len_estimated": "150", "quality_tag": "1"},
           "lo": {"measurement_id": "lo", "frag_len_estimated": "60", "quality_tag": "1"},
           "hi": {"measurement_id": "hi", "frag_len_estimated": "401", "quality_tag": "1"},
           "q": {"measurement_id": "q", "frag_len_estimated": "200", "quality_tag": "-2"},
           "edge": {"measurement_id": "edge", "frag_len_estimated": "80", "quality_tag": "-1"}}
    tab = {k: dict(v, drop_reason=SFC.fraglen_drop_reason(v)) for k, v in tab.items()}
    filled = {"measurement_id": "a", "frag_len": "210"}
    r, src, _ = SFC.with_fraglen(filled, tab)
    assert r is filled and src == "table"                     # never replaced
    r, src, why = SFC.with_fraglen({"measurement_id": "a", "frag_len": ""}, None)
    assert src == "none" and "no fragment-length table" in why
    r, src, _ = SFC.with_fraglen({"measurement_id": "a", "frag_len": ""}, tab)
    assert (r["frag_len"], src) == ("150", "estimated")
    for mid in ("lo", "hi", "q"):
        r, src, why = SFC.with_fraglen({"measurement_id": mid, "frag_len": " "}, tab)
        assert src == "none" and r["frag_len"] == " " and why
    assert SFC.with_fraglen({"measurement_id": "edge", "frag_len": ""}, tab)[1] == "estimated"
    assert SFC.with_fraglen({"measurement_id": "zz", "frag_len": ""}, tab)[1] == "none"


def test_load_fraglen_table(tmp_path):
    p = tmp_path / "fl.tsv"
    _write_tsv(p, [{"measurement_id": "a", "frag_len_estimated": "500", "quality_tag": "1"}],
               FL_COLS)
    t = SFC.load_fraglen_table(p)
    assert t["a"]["drop_reason"].startswith("implausible")
    _write_tsv(p, [{"measurement_id": "a", "x": "1"}], ["measurement_id", "x"])
    with pytest.raises(ValueError):
        SFC.load_fraglen_table(p)


# ------------------------------------------------------------------------------------- scorer

def _env(root: Path, blank=()) -> dict:
    """Synthetic tables (every ChIP row has frag_len 200); H3K4me3 s0 has an observed control;
    pipeline signal tracks = 0.7 x counts. `blank`: (cell, assay, role) whose base frag_len is
    emptied."""
    t = make_synthetic_tables(root, n_cells=8, val_tracks=VAL)
    rows = _tsv(t / "measurements.tsv")
    base = {(r["cell"], r["assay"]): r for r in rows if r["arm_knob"] == ""}
    base[("s0", "H3K4me3")]["control_measurement_id"] = base[("s0", "chipseq-control")][
        "measurement_id"]
    for (cell, a), r in base.items():     # training ChIP tracks need a control for ag2_baselines
        if a == "H3K4me3" and r["in_eic"] == "train" and (cell, "chipseq-control") in base:
            r["control_measurement_id"] = base[(cell, "chipseq-control")]["measurement_id"]
    for r in rows:
        if r["arm_knob"] == "" and (r["cell"], r["assay"], r["in_eic"]) in blank:
            r["frag_len"] = ""
    _write_tsv(t / "measurements.tsv", rows)
    for (cell, a), r in base.items():
        if a in ("H3K4me3", "DNase-seq") and r["in_eic"] == "val":
            d = Path(r["counts_path"][3:]).parent
            for nm, src in (("signal", "counts"), ("pr1_signal", "pr1"), ("pr2_signal", "pr2")):
                with h5py.File(d / f"{src}.h5", "r") as fi, h5py.File(d / f"{nm}.h5", "w") as fo:
                    for c in fi:
                        fo.create_dataset(c, data=np.asarray(fi[c][:], np.float32) * 0.7)
    ev = E.build(t, root / "evalset.json", n_regions=2, n_fit_regions=1, region_bins=RB, seed=0)
    ev["imputation_targets"] = [x for x in ev["imputation_targets"]
                                if x["assay"] in ("H3K4me3", "DNase-seq")]
    ev["correction_pairs"], ev["replicate_pairs"] = [], []
    (root / "evalset.json").write_text(json.dumps(ev))
    return {"root": root, "tables": t, "evalset": root / "evalset.json",
            "rows": {r["measurement_id"]: r for r in rows}, "base": base}


def _fl_table(env, path, value="200", roles=("val", "train")) -> Path:
    """An estimate for EVERY ChIP base row of `roles` (also rows with a table value)."""
    rows = [{"measurement_id": r["measurement_id"], "frag_len_estimated": value, "quality_tag": "1"}
            for r in env["rows"].values() if r["arm_knob"] == "" and r["in_eic"] in roles
            and r["assay"] == "H3K4me3"]
    _write_tsv(path, rows, FL_COLS)
    return path


def _score(env, out, **kw):
    from onion_eval import score as S
    return S.score(env["evalset"], out, baseline="position_average", **KW, **kw)


def _strip(obj, root):
    """scores.json without the volatile keys, the FL keys and the tmp paths."""
    drop = {"wall_seconds", "timing_seconds", "torch_threads", "evalset_sha256", "tables_sha256",
            "scorer_version", "scorer_settings", "fraglen_table", "fraglen_source", "fraglen_bp",
            "fraglen_reason", "fraglen_source_counts", "fraglen_estimated_targets", "evalset",
            "tables"}
    if isinstance(obj, dict):
        return {k: _strip(v, root) for k, v in obj.items() if k not in drop}
    if isinstance(obj, list):
        return [_strip(v, root) for v in obj]
    if isinstance(obj, str):
        return obj.replace(str(root), "<ROOT>")
    return obj


BLANK_VAL = (("s0", "H3K4me3", "val"), ("s1", "H3K4me3", "val"))
BLANK_TRAIN = (("s4", "H3K4me3", "train"), ("s5", "H3K4me3", "train"))


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    full = _env(tmp_path_factory.mktemp("full") / "env")
    blank = _env(tmp_path_factory.mktemp("blank") / "env", blank=BLANK_VAL + BLANK_TRAIN)
    out = {}
    out["full_plain"] = _score(full, full["root"] / "o_plain", dump_signal=full["root"] / "d_plain")
    out["full_fl"] = _score(full, full["root"] / "o_fl",
                            fraglen_table=_fl_table(full, full["root"] / "fl150.tsv", "150"))
    out["blank_plain"] = _score(blank, blank["root"] / "o_plain",
                                dump_signal=blank["root"] / "d_plain")
    out["blank_fl"] = _score(blank, blank["root"] / "o_fl", dump_signal=blank["root"] / "d_fl",
                             fraglen_table=_fl_table(blank, blank["root"] / "fl200.tsv", "200"))
    out["blank_bad"] = _score(blank, blank["root"] / "o_bad",
                              fraglen_table=_fl_table(blank, blank["root"] / "fl50.tsv", "50"))
    return {"full": full, "blank": blank, **out}


def _per_target(s):
    return s[H1][SIG]["per_target"]


def test_option_never_touches_a_table_value(scored):
    """Full tables + a table of WRONG estimates (150 bp for every row): nothing moves."""
    a, b = scored["full_plain"], scored["full_fl"]
    assert "fraglen_table" not in a and "fraglen_source_counts" not in a[H1][SIG]
    assert _strip(a, scored["full"]["root"]) == _strip(b, scored["full"]["root"])
    pt = _per_target(b)
    chip = [t for t, v in pt.items() if v["assay"] == "H3K4me3"]
    assert chip and all(pt[t]["fraglen_source"] == "table" for t in chip)
    assert all("fraglen_source" not in v for t, v in pt.items() if t not in chip)
    assert b[H1][SIG]["fraglen_source_counts"] == {"table": len(chip), "estimated": 0, "none": 0}
    assert b["fraglen_table"]["n_rows"] > 0 and b["fraglen_table"]["n_dropped"] == 0


def test_without_option_blanked_targets_are_undefined(scored):
    pt = _per_target(scored["blank_plain"])
    blanked = [r["measurement_id"] for k, r in scored["blank"]["base"].items()
               if k + ("val",) in BLANK_VAL]
    assert len(blanked) == 2
    for t in blanked:
        assert "frag_len" in pt[t]["lambda_reason"] and "fraglen_source" not in pt[t]


def test_estimate_equal_to_the_blanked_value_restores_the_scores(scored):
    """Second route: blanked tables + estimate 200 == the original tables (frag_len 200)."""
    full, fl = scored["full_plain"], scored["blank_fl"]
    assert _strip(full, scored["full"]["root"]) == _strip(fl, scored["blank"]["root"])
    pt = _per_target(fl)
    blanked = sorted(r["measurement_id"] for k, r in scored["blank"]["base"].items()
                     if k + ("val",) in BLANK_VAL)
    assert fl[H1][SIG]["fraglen_estimated_targets"] == blanked
    assert all(pt[t]["fraglen_source"] == "estimated" and pt[t]["fraglen_bp"] == "200"
               for t in blanked)
    assert fl[H1][SIG]["fraglen_source_counts"]["estimated"] == 2
    meta = json.loads((scored["blank"]["root"] / "d_fl" / "leave_one_assay_out" /
                       "tracks.json").read_text())
    assert {t for t, v in meta["fraglen_source"].items() if v == "estimated"} == set(blanked)


def test_implausible_estimate_is_not_used(scored):
    s = scored["blank_bad"]
    pt = _per_target(s)
    assert s[H1][SIG]["fraglen_source_counts"]["estimated"] == 0
    assert s[H1][SIG]["fraglen_source_counts"]["none"] == 2
    assert s["fraglen_table"]["n_dropped"] == s["fraglen_table"]["n_rows"]
    assert all("implausible" in v["fraglen_reason"] for v in pt.values()
               if v.get("fraglen_source") == "none")


def test_dump_writes_truth_without_prediction_signal(scored):
    env = scored["blank"]
    ev = json.loads(env["evalset"].read_text())
    regions = [(c, int(s)) for c, s in ev["regions"]]
    d = env["root"] / "d_plain" / "leave_one_assay_out"
    meta = json.loads((d / "tracks.json").read_text())
    blanked = {r["measurement_id"] for k, r in env["base"].items() if k + ("val",) in BLANK_VAL}
    assert set(meta["truth_only"]) == blanked
    assert not blanked & set(meta["targets"])
    for mid in blanked:
        assert not (d / f"{mid}.signal.npy").exists()
        cdir = Path(env["rows"][mid]["counts_path"][3:]).parent
        with h5py.File(cdir / "signal.h5", "r") as f:
            want = np.concatenate([np.asarray(f[c][s:s + RB], np.float64) for c, s in regions])
        np.testing.assert_allclose(np.load(d / f"{mid}.truth.npy"), np.arcsinh(want), rtol=1e-6)
    # with the estimate every target has its signal, and none is truth-only
    m2 = json.loads((env["root"] / "d_fl" / "leave_one_assay_out" / "tracks.json").read_text())
    assert m2["truth_only"] == [] and blanked <= set(m2["targets"])


def test_score_cli_passes_the_option(scored, monkeypatch, tmp_path):
    from onion_eval import score as S
    seen = {}
    monkeypatch.setattr(S, "score", lambda *a, **k: seen.update(k) or {"skipped": "x"})
    env = scored["blank"]
    S.main(["--baseline", "position_average", "--evalset", str(env["evalset"]), "--out",
            str(tmp_path / "o"), "--fraglen-table", "x.tsv"])
    assert seen["fraglen_table"] == "x.tsv"
    seen.clear()
    S.main(["--baseline", "position_average", "--evalset", str(env["evalset"]), "--out",
            str(tmp_path / "o2")])
    assert "fraglen_table" not in seen


# ------------------------------------------------------------------------------------- ag2_baselines

def test_ag2_baselines_training_fallback(scored, tmp_path):
    from onion_eval import ag2_baselines as AB
    full, blank = scored["full"], scored["blank"]
    tr_blank = sorted(r["measurement_id"] for k, r in blank["base"].items()
                      if k + ("train",) in BLANK_TRAIN)
    # without the table: the blanked training tracks are left out and counted, not fatal
    o1 = tmp_path / "plain"
    AB.run(blank["tables"], blank["evalset"], blank["root"] / "d_plain", o1, log=lambda *a: None)
    m1 = json.loads((o1 / AB.MEAN / "leave_one_assay_out" / "tracks.json").read_text())
    for mid in tr_blank:
        assert m1["training_tracks_left_out"][mid] == "ChIP-seq training track with no fragment length"
    # the truth-only targets of the dump are targets here (they have a truth)
    d_meta = json.loads((blank["root"] / "d_plain" / "leave_one_assay_out" /
                         "tracks.json").read_text())
    assert set(d_meta["truth_only"]) <= set(m1["targets"])
    # with the table (estimate 200 = the blanked value): the training mean equals the full tables'
    o2 = tmp_path / "fl"
    AB.run(blank["tables"], blank["evalset"], blank["root"] / "d_plain", o2, log=lambda *a: None,
           fraglen_table=blank["root"] / "fl200.tsv")
    o3 = tmp_path / "full"
    AB.run(full["tables"], full["evalset"], full["root"] / "d_plain", o3, log=lambda *a: None)
    m2 = json.loads((o2 / AB.MEAN / "leave_one_assay_out" / "tracks.json").read_text())
    m3 = json.loads((o3 / AB.MEAN / "leave_one_assay_out" / "tracks.json").read_text())
    assert not set(tr_blank) & set(m2["training_tracks_left_out"])
    assert {m2["training_tracks_fraglen_source"][t] for t in tr_blank} == {"estimated"}
    assert "training_tracks_fraglen_source" not in m1
    assert m2["n_training_tracks"] == m3["n_training_tracks"]
    for mid in m3["targets"]:
        a = np.load(o2 / AB.MEAN / "leave_one_assay_out" / f"{mid}.signal.npy")
        b = np.load(o3 / AB.MEAN / "leave_one_assay_out" / f"{mid}.signal.npy")
        np.testing.assert_array_equal(a, b)
