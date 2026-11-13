"""AGc: the same-track scorer of the AG2 comparison (tools/onion_eval/ag2_score.py).

Synthetic method track files (pinned interface F) on 4 regions of 1,024 bins: 6 validation targets of
one mark in 6 cells (4 of status `unseen`, 2 of status `observed_matched`), 3 training tracks for the
ENCODE peak tile set, and one directory per method kind."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import ag2_score as AS  # noqa: E402
from onion_eval import evalset as E  # noqa: E402
from onion_eval.ag2_baselines import regions_sha256  # noqa: E402

RB = 1024
REGIONS = [("chr1", 0), ("chr1", 2000), ("chr2", 0), ("chr2", 3000)]
N = RB * len(REGIONS)
REGIME = "leave_one_assay_out"
ASSAY = "H3K4me3"
STATUS = {"c1": "unseen", "c2": "unseen", "c3": "unseen", "c4": "unseen",
          "c5": "observed_matched", "c6": "observed_matched"}


def _tsv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _peaks_file(path, mask):
    lines = []
    for i, (c, sb) in enumerate(REGIONS):
        for j in np.flatnonzero(mask[i * RB:(i + 1) * RB]):
            s = (sb + int(j)) * 25
            lines.append(f"{c}\t{s}\t{s + 25}\tp\t0\t.\t1\t1\t1\t1\n")
    Path(path).write_text("".join(lines))


def _method(root, name, kind, split, arrays, ev_sha, regions=REGIONS, extra=None):
    d = root / name / REGIME
    d.mkdir(parents=True)
    for fn, a in arrays.items():
        np.save(d / fn, np.asarray(a, np.float32))
    targets = sorted({fn.split(".")[0] for fn in arrays})
    meta = {"method": name, "kind": kind, "evalset_sha256": ev_sha,
            "regions_sha256": regions_sha256(regions, RB), "split": split, "regime": REGIME,
            "region_bins": RB, "n_regions": len(regions), "lambda_source": "observed_control",
            "targets": targets, **(extra or {})}
    (d / "tracks.json").write_text(json.dumps(meta))
    return root / name


def make_world(root: Path, split: str) -> dict:
    rng = np.random.default_rng(1)
    root.mkdir(parents=True, exist_ok=True)
    tdir = root / "tables"
    tdir.mkdir()
    cells = sorted(STATUS)
    rows, audit, items = [], [], []
    for k, c in enumerate(cells):
        mid = f"m_{c}"
        rows.append({"measurement_id": mid, "eic_track": f"T{c}", "in_eic": split, "cell": c,
                     "assay": ASSAY, "assay_class": "histone", "arm_knob": "", "layer": "uniform",
                     "unit": "pooled"})
        audit.append({"eic_track": f"T{c}", "role": split, "cell": c, "assay": ASSAY,
                      "status": STATUS[c], "api_label": "observed" if STATUS[c] != "unseen"
                      else "imputed_supported"})
        items.append({"target": mid, "inputs": [], "regime": REGIME, "split": split,
                      "assay": ASSAY, "cell": c})
    for c in ("c7", "c8", "c9"):
        rows.append({"measurement_id": f"m_{c}", "eic_track": f"T{c}", "in_eic": "train", "cell": c,
                     "assay": ASSAY, "assay_class": "histone", "arm_knob": "", "layer": "uniform",
                     "unit": "pooled"})
    _tsv(tdir / "measurements.tsv", rows)
    _tsv(root / "audit.tsv", audit)
    evd = root / "ev"
    evd.mkdir()
    ev_path = evd / ("evalset_blind.json" if split == "blind" else "evalset.json")
    ev_path.write_text(json.dumps({"regions": [list(r) for r in REGIONS], "region_bins": RB,
                                   "imputation_targets": items}))
    ev_sha = E.sha256_file(ev_path)
    shared = np.convolve(rng.normal(0, 1, N), np.ones(9) / 3, "same")
    a = np.arcsinh(np.exp(shared))
    truth, half_sig, half_truth = {}, {}, {}
    peaks = root / "peaks"
    peaks.mkdir()
    for c in cells:
        own = np.convolve(rng.normal(0, 1, N), np.ones(5) / 2, "same")
        y = np.arcsinh(np.exp(0.6 * shared + own)).astype(np.float32)
        truth[f"m_{c}"] = y
        h = (y + rng.normal(0, 0.3, N)).clip(0).astype(np.float32)
        half_sig[f"m_{c}"] = h
        half_truth[f"m_{c}"] = h          # the half row's own truth: identical to its signal
        _peaks_file(peaks / f"T{c}.narrowPeak", y >= AS.PEAK_CALL)
    for c in ("c7", "c8", "c9"):
        _peaks_file(peaks / f"T{c}.narrowPeak", rng.random(N) < 0.05)
    md = root / "methods"
    md.mkdir()
    m = {}
    m["training_mean_signal"] = _method(md, "training_mean_signal", "oracle_baseline", split,
                                        {f"{t}.signal.npy": a for t in truth}, ev_sha)
    seeds = []
    for s in range(2):    # two seeds: same regions, different eval-set file hash
        seeds.append(_method(md, f"final_s{s}", "run", split,
                             {**{f"{t}.signal.npy": y + rng.normal(0, 0.5, N) for t, y in truth.items()},
                              **{f"{t}.truth.npy": y for t, y in truth.items()},
                              **{f"{t}.peakprob.npy": (y >= AS.PEAK_CALL) * 0.9 for t, y in truth.items()}},
                             f"{ev_sha[:-4]}{s:04d}"))
    m["minicandi_final"] = seeds
    m["planted_monotone"] = _method(md, "planted_monotone", "run", split,
                                    {**{f"{t}.signal.npy": np.exp(2 * y) for t, y in truth.items()},
                                     **{f"{t}.truth.npy": y for t, y in truth.items()}}, ev_sha)
    m["pseudoreplicate_half"] = _method(md, "pseudoreplicate_half", "replicate_floor", split,
                                        {**{f"{t}.signal.npy": v for t, v in half_sig.items()},
                                         **{f"{t}.truth.npy": v for t, v in half_truth.items()}},
                                        ev_sha)
    ag2 = root / "ag2pack" / REGIME
    ag2.mkdir(parents=True)
    for t, y in truth.items():
        np.save(ag2 / f"{t}.matched.signal.npy", np.exp(y + rng.normal(0, 0.4, N)).astype(np.float32))
        np.save(ag2 / f"{t}.otherend.signal.npy", np.exp(y + rng.normal(0, 0.6, N)).astype(np.float32))
    (ag2 / "EVALUATION_ONLY").write_text("")
    (ag2 / "tracks.json").write_text(json.dumps({
        "method": "ag2", "kind": "ag2", "evalset_sha256": ev_sha,
        "regions_sha256": regions_sha256(REGIONS, RB), "split": split, "regime": REGIME,
        "region_bins": RB, "n_regions": len(REGIONS), "lambda_source": "none",
        "targets": sorted(truth)}))
    return {"root": root, "ev": ev_path, "tables": tdir, "audit": root / "audit.tsv",
            "truth_from": seeds[0], "methods": m, "ag2": root / "ag2pack", "peaks": peaks,
            "truth": truth, "ev_sha": ev_sha, "md": md}


def _specs(w, extra=()):
    out = []
    for name, d in w["methods"].items():
        ds = d if isinstance(d, list) else [d]
        out.append(f"{name}=" + ",".join(str(x) for x in ds))
    return out + list(extra)


def _run(w, out, **kw):
    split = kw.pop("split", "val")
    return AS.score(w["ev"], w["audit"], w["tables"], w["truth_from"], kw.pop("methods", _specs(w)),
                    w["ag2"], w["peaks"], out, split=split, n_boot=200, log=lambda *_: None, **kw)


def _read(path):
    return E.read_tsv(path)


@pytest.fixture(scope="module")
def val(tmp_path_factory):
    root = tmp_path_factory.mktemp("agc")
    w = make_world(root / "w", "val")
    _run(w, root / "out")
    w["out"] = root / "out"
    return w


def test_outputs_written(val):
    for f in ("per_target.tsv", "ag2_board_val.tsv", "ag2_contrasts_val.tsv", "SCORED"):
        assert (val["out"] / f).is_file()


def test_monotone_transform_scores_one_after_recalibration(val):
    rows = [r for r in _read(val["out"] / "per_target.tsv")
            if r["method"] == "planted_monotone" and r["truth_peaks"] == "encode"]
    assert len(rows) == 6
    for r in rows:
        assert float(r["rspec_cal"]) > 0.99
        assert float(r["pearson_cal"]) > 0.99
        assert float(r["pearson"]) < 0.95
        assert float(r["spearman"]) > 0.999


def test_ag2_trained_track_only_in_labelled_contrast(val):
    per = _read(val["out"] / "per_target.tsv")
    for r in per:
        if r["cell"] in ("c5", "c6"):
            assert r["track_set"] == "ag2_trained"
            assert r["comparison_label"].startswith("ag2_trained_on:")
        else:
            assert r["track_set"] == "unseen_by_both"
            assert r["comparison_label"].startswith("headline:")
    con = _read(val["out"] / "ag2_contrasts_val.tsv")
    head = [r for r in con if r["track_set"] == "unseen_by_both" and r["assay_class"] == "all"]
    lab = [r for r in con if r["track_set"] == "ag2_trained" and r["assay_class"] == "all"]
    assert head and lab
    assert all(r["n_tracks"] == "4" for r in head if r["metric"] == "rspec_cal")
    assert all(r["n_tracks"] == "2" for r in lab if r["metric"] == "rspec_cal")
    assert all(r["comparison_label"].startswith("headline:") for r in head)
    assert all(r["comparison_label"].startswith("ag2_trained_on:") for r in lab)
    assert all(r["ci_lo"] == AS.TOO_FEW for r in head + lab)   # 4 and 2 tracks: under the minimum


def test_seed_dirs_with_other_file_hash_accepted_other_regions_refused(val, tmp_path):
    seeds = val["methods"]["minicandi_final"]
    metas = [json.loads((s / REGIME / "tracks.json").read_text()) for s in seeds]
    assert metas[0]["evalset_sha256"] != metas[1]["evalset_sha256"]
    rows = [r for r in _read(val["out"] / "per_target.tsv") if r["method"] == "minicandi_final"]
    assert rows and all(r["n_seeds"] == "2" for r in rows)
    other = [("chr1", 0), ("chr1", 2000), ("chr2", 0), ("chr2", 3001)]
    bad = _method(tmp_path, "shifted", "run", "val",
                  {f"{t}.signal.npy": y for t, y in val["truth"].items()}, val["ev_sha"],
                  regions=other)
    with pytest.raises(AS.Refused, match="regions_sha256"):
        _run(val, tmp_path / "o", methods=_specs(val, [f"shifted={bad}"]))


def test_training_mean_scores_rspec_zero(val):
    rows = [r for r in _read(val["out"] / "per_target.tsv") if r["method"] == "training_mean_signal"]
    assert rows and all(float(r["rspec_cal"]) == 0.0 and float(r["rspec"]) == 0.0 for r in rows)


def test_half_row_scored_against_its_own_truth(val):
    rows = [r for r in _read(val["out"] / "per_target.tsv") if r["method"] == "pseudoreplicate_half"]
    assert len(rows) == 12
    for r in rows:      # signal equals its own truth, not the pooled truth
        assert float(r["pearson"]) == pytest.approx(1.0, abs=1e-6)


def test_labels_on_every_row(val):
    for f in ("per_target.tsv", "ag2_board_val.tsv"):
        rows = _read(val["out"] / f)
        assert rows
        for r in rows:
            for k in AS.LABEL_COLS + ["comparison_label", "trained_on_track"]:
                assert r[k] != "", (f, k, r["method"])
    board = _read(val["out"] / "ag2_board_val.tsv")
    ag = [r for r in board if r["method"] == "ag2_matched"]
    assert {r["reads_target_cell_inputs_at_test_positions"] for r in ag} == {"no"}
    assert {r["nbpeak_f1"] for r in ag} == {AS.NA_APPL}
    fin = [r for r in board if r["method"] == "minicandi_final"]
    assert all(r["nbpeak_f1"] not in (AS.NA_APPL, AS.NA_AVAIL) for r in fin)
    assert {r["trained_on_track"] for r in ag if r["track_set"] == "ag2_trained"} == {"yes"}
    assert {r["trained_on_track"] for r in ag if r["track_set"] == "unseen_by_both"} == {"no"}


def test_differing_truth_refused(val, tmp_path):
    bad = _method(tmp_path, "badtruth", "run", "val",
                  {**{f"{t}.signal.npy": y for t, y in val["truth"].items()},
                   **{f"{t}.truth.npy": y + 0.01 for t, y in val["truth"].items()}}, val["ev_sha"])
    with pytest.raises(AS.Refused, match="truth of"):
        _run(val, tmp_path / "o", methods=_specs(val, [f"badtruth={bad}"]))


def test_target_in_two_track_sets_refused(val, tmp_path):
    rows = _read(val["audit"])
    rows.append(dict(rows[0], status="observed_matched"))   # unseen and observed for one track
    w = dict(val, audit=tmp_path / "audit2.tsv")
    _tsv(w["audit"], rows)
    with pytest.raises(AS.Refused, match="two track sets"):
        _run(w, tmp_path / "o")


def test_missing_scored_target_refused(val, tmp_path):
    t0 = sorted(val["truth"])[0]
    bad = _method(tmp_path, "partial", "run", "val",
                  {f"{t}.signal.npy": y for t, y in val["truth"].items() if t != t0}, val["ev_sha"])
    with pytest.raises(AS.Refused, match="lacks"):
        _run(val, tmp_path / "o", methods=_specs(val, [f"partial={bad}"]))


def test_second_blind_pass_refused(tmp_path):
    w = make_world(tmp_path / "wb", "blind")
    with pytest.raises(PermissionError):
        _run(w, tmp_path / "o0", split="blind")
    _run(w, tmp_path / "o1", split="blind", allow_blind=True)
    led = _read(w["ev"].parent / "BLIND_PASSES.tsv")
    assert [r["status"] for r in led] == ["started", "scored"]
    assert led[0]["run"] == f"ag2:{w['ag2'].resolve()}"
    with pytest.raises(AS.Refused, match="repeat-blind-reason"):
        _run(w, tmp_path / "o2", split="blind", allow_blind=True)
    assert len(_read(w["ev"].parent / "BLIND_PASSES.tsv")) == 2
    _run(w, tmp_path / "o3", split="blind", allow_blind=True, repeat_blind_reason="a test")
    assert len(_read(w["ev"].parent / "BLIND_PASSES.tsv")) == 4
    assert (tmp_path / "o3" / "ag2_board_blind.tsv").is_file()


def test_bootstrap_interval_of_constant_difference_excludes_zero():
    rng = np.random.default_rng(3)
    d = 0.05 + rng.normal(0, 0.01, 12)
    cells = [f"c{i % 4}" for i in range(12)]
    lo, hi = AS.cluster_bootstrap(d, cells, 2000)
    assert 0 < lo <= d.mean() <= hi
    lo, hi = AS.cluster_bootstrap(np.full(12, 0.05), cells, 2000)
    assert lo == pytest.approx(0.05) and hi == pytest.approx(0.05)


def test_contrast_interval_when_set_is_large_enough():
    per = []
    for i in range(12):
        base = {"target": f"t{i}", "cell": f"c{i % 4}", "assay_class": "histone",
                "track_set": "unseen_by_both", "truth_peaks": "encode"}
        for m, v in (("ag2_matched", 0.5), ("minicandi_final", 0.6)):
            per.append(dict(base, method=m, **{k: v for k in AS.CONTRAST_METRICS}))
    con = [r for r in AS._contrasts(per, "headline", 500)
           if r["metric"] == "rspec_cal" and r["assay_class"] == "all"]
    assert len(con) == 1 and con[0]["ci_lo"] > 0 and con[0]["call"] == "minicandi_final ahead"
    assert con[0]["n_ahead"] == 12


def test_module_does_not_import_alphagenome():
    assert "alphagenome" not in sys.modules


class _FakeStore:
    """The slice of candi.store.reader.CorpusStore that the gate reads."""

    def __init__(self, tracks, nb):
        self.tracks, self.nb = tracks, nb
        self.genome = self

    def n_bins(self, c):
        return self.nb[c]

    def mask(self, c, s, e):
        m = np.ones(e - s, bool)
        m[:7] = False
        return m

    def __getitem__(self, b):
        st = self

        class _B:
            def pval(self, c, s, e, assays):
                return np.sinh(st.tracks[(b, assays[0], c)])[s:e, None]
        return _B()


def test_gate_scores_max_layout_and_compares(tmp_path):
    rng = np.random.default_rng(5)
    nb = {"chr20": 1280, "chr21": 640}
    bios = ["b1", "b2", "b3_grp1_rep1", "b3_grp1_rep2"]
    tracks, rows = {}, []
    for b in bios:
        for c, n in nb.items():
            tracks[(b, ASSAY, c)] = np.abs(rng.normal(1, 1, n))
        rows.append({"biosample": b, "assay": ASSAY, "hidden": "False" if b != "b1" else "True",
                     "exp_accession": f"E{b[:2]}"})
    _tsv(tmp_path / "tracks.tsv", rows)
    _tsv(tmp_path / "targets.tsv", [{"biosample": "b1", "assay": ASSAY}])
    for c, n in nb.items():
        (tmp_path / "pred" / c).mkdir(parents=True)
        np.savez(tmp_path / "pred" / c / f"b1__{ASSAY}.npz",
                 matched=np.abs(rng.normal(1, 1, -(-n * 25 // 128))))
    (tmp_path / "tiles").mkdir()
    for b in bios:
        np.savez(tmp_path / "tiles" / f"{b}.npz", tracks=np.array([ASSAY]),
                 **{f"cnt_{c}": (rng.random((n // 20, 1)) < 0.3).astype(int) for c, n in nb.items()})
    ref = tmp_path / "ref.tsv"
    _tsv(ref, [{"method": "ag2", "setting": "all", "stratum": "all", "biosample": "b1",
                "assay": ASSAY, "pearson_cal": "nan", "spearman": "nan", "auprc": "nan",
                "rspec_cal": "nan"}])
    store = _FakeStore(tracks, nb)
    kw = dict(tracks=tmp_path / "tracks.tsv", chroms=tuple(nb), store_obj=store, log=lambda *_: None)
    assert not AS.gate(tmp_path / "targets.tsv", tmp_path / "pred", None, tmp_path / "tiles", ref,
                       tmp_path / "g0", **kw)
    ours = {r["metric"]: r["ours"] for r in _read(tmp_path / "g0" / "gate.tsv")}
    assert set(ours) == {"pearson_cal", "spearman", "auprc", "rspec_cal"}
    _tsv(ref, [{"method": "ag2", "setting": "all", "stratum": "all", "biosample": "b1",
                "assay": ASSAY, **ours}])
    assert AS.gate(tmp_path / "targets.tsv", tmp_path / "pred", None, tmp_path / "tiles", ref,
                   tmp_path / "g1", **kw)
