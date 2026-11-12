"""Tests of tools/onion_data/other_experiments.py (t257, chunk OXs): the selection checks, the order
rule, the one-track-per-experiment assignment, fragments for paired-end candidates, and the pair
smoke. Hand-written catalog fixtures; no network (the portal and HEAD answers are passed in)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "tools",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import build_list as B  # noqa: E402
from onion_data import other_experiments as OX  # noqa: E402

M = 1_000_000
TERM = "liver"
LAB_E, LAB_O = "Lab E", "Lab O"


class Cat:
    """A tiny ENCODE catalog in memory, written with the D1 column names."""

    def __init__(self):
        self.exps, self.libs, self.files = [], [], []
        self.n = 0

    def exp(self, acc, lab=LAB_E, donor="D1", term=TERM, reads=10 * M, pe=False, lib=None,
            controls=""):
        self.exps.append({"experiment": acc, "biosample_term": term, "lab": lab,
                          "status": "released", "control_experiments": controls})
        lib = lib or f"LIB_{acc}"
        self.libs.append({"experiment": acc, "library": lib, "biosample": f"BS_{acc}",
                          "donor": donor, "bio_rep": "1", "tech_rep": "1"})
        self.n += 1
        f1 = f"FF{self.n:04d}A"
        if pe:
            f2 = f"FF{self.n:04d}B"
            for f, mate, end in ((f1, f2, "1"), (f2, f1, "2")):
                self.files.append(self._f(f, acc, lib, "paired-ended", mate, end, reads))
        else:
            self.files.append(self._f(f1, acc, lib, "single-ended", "", "", reads))

    @staticmethod
    def _f(f, acc, lib, rt, mate, end, reads):
        return {"file": f, "experiment": acc, "library": lib, "file_format": "fastq",
                "run_type": rt, "read_length": "36" if rt == "single-ended" else "100",
                "platform": "HiSeq", "paired_with": mate, "bio_reps": "1", "tech_reps": "1_1",
                "status": "released", "paired_end": end, "read_count": str(reads),
                "file_size": "1000000000", "url": ""}

    def write(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        B.write_tsv(d / "experiments.tsv", ("experiment", "biosample_term", "lab", "status",
                                            "control_experiments"), self.exps)
        B.write_tsv(d / "libraries.tsv", ("experiment", "library", "biosample", "donor",
                                          "bio_rep", "tech_rep"), self.libs)
        B.write_tsv(d / "files.tsv", tuple(self.files[0]), self.files)
        return d


def track(k, split, eic, cands, cls="narrow_histone"):
    return {"eic_track": k, "split": split, "cell": "liver", "encode_term": TERM,
            "assay": "H3K4me3", "assay_class": cls, "eic_experiment": eic, "eic_lab": LAB_E,
            "best_candidate": cands[0], "other_accessions": ";".join(cands)}


@pytest.fixture
def world(tmp_path):
    c = Cat()
    # EIC experiments: four tracks plus a training experiment that only sets the depth floor
    for e in ("E1", "E4", "E5", "E6"):
        c.exp(e, reads=10 * M, lib="LIB_SHARED" if e == "E1" else None)
    c.exp("E2", reads=10 * M)
    c.exp("A", lab=LAB_O, donor="D1", reads=20 * M, controls="CTL_A")   # other lab, same donor
    c.exp("B", lab=LAB_O, donor="D9", reads=10 * M)                     # other lab, other donor
    c.exp("C", lab=LAB_E, donor="D1", reads=10 * M)                     # same lab, same donor
    c.exp("P", lab=LAB_O, donor="D1", reads=10 * M)                     # perturbed (portal)
    c.exp("S", lab=LAB_O, donor="D1", reads=1 * M)                      # shallow
    c.exp("SL", lab=LAB_O, donor="D1", reads=10 * M, lib="LIB_SHARED")  # shares E1's library
    c.exp("X1", lab=LAB_O, donor="D1", reads=10 * M)
    c.exp("X2", lab=LAB_O, donor="D9", reads=10 * M)
    c.exp("PE_OK", lab=LAB_O, donor="D1", reads=10 * M, pe=True)        # 20 M reads, 10 M fragments
    c.exp("PE_LOW", lab=LAB_O, donor="D1", reads=6 * M, pe=True)        # 12 M reads, 6 M fragments
    c.exp("CTL_A", lab=LAB_O, donor="D1", reads=10 * M)
    cat_dir = c.write(tmp_path / "catalog")
    census = [
        track("C09M01", "val", "E1", ["B", "C", "A", "P", "S", "SL", "E2"]),
        track("C01M01", "train", "E5", ["X1", "X2"]),    # sorts first by name, but is training
        track("C05M01", "val", "E4", ["X2", "X1"]),
        track("C07M01", "train", "E6", ["PE_LOW", "PE_OK"]),
        {**track("C08M01", "train", "E6", ["Z"]), "best_candidate": "", "other_accessions": ""},
    ]
    ledger = [{"eic_track": t, "role": r, "assay_class": "narrow_histone", "experiment": e,
               "encode_possible_controls": ""}
              for t, r, e in (("C09M01", "val", "E1"), ("C01M01", "train", "E5"),
                              ("C05M01", "val", "E4"), ("C07M01", "train", "E6"),
                              ("C02M01", "train", "E2"))]
    build = [{"experiment": e, "role": r, "arm": "base", "unit": "pooled",
              "assay_class": "narrow_histone", "control_experiment": ""}
             for e, r in (("E1", "val"), ("E4", "val"), ("E5", "train"), ("E6", "train"),
                          ("E2", "train"))]
    # the floor: fragments of the training base experiments E5, E6, E2 = 10 M each
    ok = {"status": "released", "errors": [], "perturbed": False, "treated": False,
          "modified": False, "term": TERM}
    portal = {a: dict(ok) for a in ("A", "B", "C", "S", "SL", "E2", "X1", "X2", "PE_OK", "PE_LOW")}
    portal["P"] = {**ok, "perturbed": True, "treated": True}
    wanted = {r["experiment"] for r in c.exps}
    cat = OX.load_catalog(cat_dir, wanted)
    probe = {f["file"]: "200" for f in c.files}
    return {"census": census, "cat": cat, "cat_dir": cat_dir, "ledger": ledger, "build": build,
            "portal": portal, "probe": probe, "tmp": tmp_path}


def run(w):
    return OX.select(w["census"], w["cat"], w["ledger"], w["build"], w["portal"], w["probe"])


def ev(res, track, acc):
    return next(c for c in res["evals"][track] if c["experiment"] == acc)


def test_refusals(world):
    res = run(world)
    assert any(f.startswith("2 perturbed") for f in ev(res, "C09M01", "P")["fails"])
    assert any(f.startswith("6 fragments") for f in ev(res, "C09M01", "S")["fails"])
    assert any(f.startswith("3 shares library LIB_SHARED") for f in ev(res, "C09M01", "SL")["fails"])
    assert "3 EIC ledger experiment" in ev(res, "C09M01", "E2")["fails"]
    for a in ("A", "B", "C"):
        assert ev(res, "C09M01", a)["ok"], ev(res, "C09M01", a)["fails"]


def test_order_other_lab_same_donor_first(world):
    rows = {r["eic_track"]: r for r in run(world)["rows"]}
    r = rows["C09M01"]
    # B is depth-closer (log2 ratio 0 against 1) but has another donor; C is the same lab
    assert r["experiment"] == "A" and r["pair_class"] == "experiment;lab"
    assert r["fallback"] == "B" and r["fallback_pair_class"] == "experiment;lab;donor"
    assert r["own_control"] == "CTL_A" and r["own_control_state"] == "CTL_A:served"
    assert r["log2_depth_ratio"] == "1.000" and r["checks"].startswith("1:ok;2:ok;3:ok;4:ok")


def test_shared_candidates_validation_first(world):
    res = run(world)
    rows = {r["eic_track"]: r for r in res["rows"]}
    assert rows["C05M01"]["experiment"] == "X1"      # the validation track takes the best one
    assert rows["C01M01"]["experiment"] == "X2"      # the training track gets the other
    used = [r["experiment"] for r in res["rows"]] + [r["fallback"] for r in res["rows"] if r["fallback"]]
    assert len(used) == len(set(used))


def test_paired_end_counted_in_fragments(world):
    res = run(world)
    ok, low = ev(res, "C07M01", "PE_OK"), ev(res, "C07M01", "PE_LOW")
    assert ok["fragments"] == str(10 * M) and ok["catalog_reads"] == str(20 * M)
    assert ok["log2_depth_ratio"] == "0.000" and ok["ok"]
    assert ok["tech_diff"] == "run_type;read_length"
    # 12 M reads would pass a 10 M floor; 6 M fragments do not
    assert low["catalog_reads"] == str(12 * M) and not low["ok"]
    assert any(f.startswith("6 fragments 6000000 < floor 10000000") for f in low["fails"])
    assert res["floors"]["narrow_histone"] == (10 * M, 3)


def test_dropped_and_no_candidate_track(world):
    w = dict(world)
    w["portal"] = {k: v for k, v in world["portal"].items() if k not in ("X1", "X2")}
    res = run(w)
    dropped = {d["eic_track"]: d for d in res["dropped"]}
    assert set(dropped) == {"C01M01", "C05M01"}
    assert "X1: 1 not on the portal" in dropped["C05M01"]["candidates"]
    keys = {r["eic_track"] for r in res["rows"]} | set(dropped)
    assert "C08M01" not in keys                       # a track with no census candidate


def test_cli_select_end_to_end(world, monkeypatch):
    w, t = world, world["tmp"]
    monkeypatch.setattr(OX, "portal_meta", lambda accs: w["portal"])
    monkeypatch.setattr(OX, "head_probe", lambda files, threads=16: w["probe"])
    B.write_tsv(t / "census.tsv", tuple(w["census"][0]), w["census"])
    B.write_tsv(t / "ledger.tsv", tuple(w["ledger"][0]), w["ledger"])
    B.write_tsv(t / "build.tsv", tuple(w["build"][0]), w["build"])
    assert OX.main(["select", "--census", str(t / "census.tsv"), "--catalog", str(w["cat_dir"]),
                    "--ledger", str(t / "ledger.tsv"), "--build-list", str(t / "build.tsv"),
                    "--out", str(t / "out" / "other_experiments.tsv"),
                    "--dropped", str(t / "out" / "dropped.tsv")]) == 0
    rows = B.read_tsv(t / "out" / "other_experiments.tsv")
    assert tuple(rows[0]) == OX.OUT_COLUMNS and len(rows) == 4
    assert [r["eic_track"] for r in rows] == ["C05M01", "C09M01", "C01M01", "C07M01"]
    md = (t / "out" / "SELECTION.md").read_text()
    assert "undercount" in md and "| narrow_histone | 3 | 10,000,000 |" in md


def test_pair_smoke(tmp_path):
    import h5py
    import numpy as np
    rng = np.random.default_rng(0)
    lam = rng.uniform(1, 20, 4096)
    a, b = rng.poisson(lam), rng.poisson(2 * lam)
    for name, x, depth in (("ma", a, 1000), ("mb", b, 2000), ("mc", b, 1000)):
        d = tmp_path / name
        d.mkdir()
        with h5py.File(d / "counts.h5", "w") as f:
            f.create_dataset("chr1", data=x.astype(np.int32))
            f.attrs["depth_reads"] = depth
    ev_ = {"regions": [["chr1", 0], ["chr1", 2048]], "region_bins": 2048}
    (tmp_path / "ev.json").write_text(json.dumps(ev_))
    out = tmp_path / "smoke.md"
    OX.main(["pair-smoke", "--pairs", f"{tmp_path/'ma'}:{tmp_path/'mb'},{tmp_path/'ma'}:{tmp_path/'mc'}",
             "--evalset", str(tmp_path / "ev.json"), "--out", str(out)])
    res = OX.pair_smoke([(str(tmp_path / "ma"), str(tmp_path / "mb")),
                         (str(tmp_path / "ma"), str(tmp_path / "mc"))], ev_)
    assert not res[0]["outside"] and abs(res[0]["ratio"] - 1) < 0.05   # declared depths are right
    assert res[1]["outside"] and res[1]["ratio"] < 0.6                  # mc is declared too shallow
    assert "1 of 2 pairs are outside" in out.read_text()


def test_blind_track_never_selected(world):
    w = dict(world)
    w["census"] = world["census"] + [track("C00M01", "blind", "E1", ["C", "B"])]
    res = run(w)
    assert all(r["split"] != "blind" for r in res["rows"])
    d = {x["eic_track"]: x for x in res["dropped"]}
    assert d["C00M01"]["reason"] == OX.BLIND_REASON
    assert "C: passes; blind track" in d["C00M01"]["candidates"]
    # the blind track holds nothing: the validation track C09M01 keeps its choice and fallback
    r = {x["eic_track"]: x for x in res["rows"]}["C09M01"]
    assert (r["experiment"], r["fallback"]) == ("A", "B")
