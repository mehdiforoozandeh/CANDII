"""AGk: peak files, tiles, consensus AUPRC (tools/onion_eval/ag2_peaks.py; plan Amendment 5 B.3).

Hand cases with known answers; the ENCODE portal is a fake opener (no network)."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import ag2_peaks as P  # noqa: E402

# 6 consensus tiles, positives at ranks 1 and 3: AP = 0.5 * 1 + 0.5 * 2/3
HAND_AP = 0.5 * 1.0 + 0.5 * (2.0 / 3.0)


def test_hand_case_six_tiles():
    y = np.array([1, 0, 1, 0, 0, 0], bool)
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    a, pf = P.auprc(y, np.ones(6, bool), s)
    assert a == pytest.approx(HAND_AP)
    assert pf == pytest.approx(2 / 6)


def test_background_tiles_never_enter():
    y = np.array([1, 0, 1, 0, 0, 0, 1, 0], bool)          # a target call off the consensus too
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 5.0, 9.0])  # background ranked first
    c = np.array([1, 1, 1, 1, 1, 1, 0, 0], bool)
    a, pf = P.auprc(y, c, s)
    assert a == pytest.approx(HAND_AP)
    assert pf == pytest.approx(2 / 6)
    p, r, f = P.binary_prf(y, c, np.array([1, 1, 0, 0, 0, 0, 1, 1], bool))
    assert (p, r) == (pytest.approx(0.5), pytest.approx(0.5)) and f == pytest.approx(0.5)


def test_all_positive_is_undefined_not_one():
    a, pf = P.auprc(np.ones(5, bool), np.ones(5, bool), np.arange(5.0))
    assert math.isnan(a) and pf == 1.0
    a0, pf0 = P.auprc(np.zeros(5, bool), np.ones(5, bool), np.arange(5.0))
    assert math.isnan(a0) and pf0 == 0.0


def test_frequency_bins_partition_consensus():
    rng = np.random.default_rng(0)
    calls = {f"t{i}": rng.random(400) < 0.1 * (i + 1) for i in range(5)}
    calls["empty"] = np.zeros(400, bool)                 # calls nothing: not counted
    cons, freq = P.consensus(calls)
    assert np.array_equal(cons, np.any(list(calls.values()), axis=0))
    assert set(np.round(freq[cons] * 5).astype(int)) <= {1, 2, 3, 4, 5}
    rows = P.auprc_by_frequency(calls["t2"], cons, rng.random(400), freq, n_bins=8)
    assert len(rows) == 8 and sum(r["n_tiles"] for r in rows) == int(cons.sum())
    fb = P.frequency_bin(freq, 8)[cons]
    assert [int((fb == k).sum()) for k in range(8)] == [r["n_tiles"] for r in rows]
    assert P.frequency_bin(np.array([0.125, 0.13, 1.0]), 8).tolist() == [0, 1, 7]


def test_tiles_per_region_drop_tail():
    x = np.zeros(2 * 44, bool)
    x[41] = True                                         # region 0 tail: in no tile
    x[44 + 25] = True                                    # region 1, tile 1
    assert P.tile_any(x, 20, region_bins=44).tolist() == [False, False, False, True]
    v = np.arange(88.0)
    assert P.tile_mean(v, 20, region_bins=44)[2] == pytest.approx(np.mean(np.arange(44, 64)))
    assert P.tile_max(v, 20, region_bins=44)[3] == 83.0
    assert P.derived_peaks(np.array([1.99, 2.0, 3.0])).tolist() == [False, True, True]


@pytest.mark.parametrize("gz", [False, True])
def test_rasterize_three_lines(tmp_path, gz):
    text = ("chr1\t10\t30\tp1\t0\t.\t5\t4\t3\t10\n"       # bins 0, 1
            "chr1\t199\t260\tp2\t0\t.\t5\t4\t3\t10\n"     # bins 7..10 -> region 0 bin 7 only
            "chr2\t125\t150\tp3\t0\t.\t5\t4\t3\t10\n")    # bin 5 -> region 1 (start 4) bin 1
    p = tmp_path / ("a.narrowPeak.gz" if gz else "a.narrowPeak")
    p.write_bytes(gzip.compress(text.encode()) if gz else text.encode())
    m = P.rasterize(p, [("chr1", 0), ("chr2", 4)], 8)
    want = np.zeros(16, bool)
    want[[0, 1, 7, 9]] = True
    assert m.tolist() == want.tolist()
    assert P.count_peaks(p) == 3


# ------------------------------------------------------------------------------- fetch with a fake portal

NP = gzip.compress(b"chr1\t100\t200\tx\t0\t.\t1\t1\t1\t50\nchr1\t300\t400\tx\t0\t.\t1\t1\t1\t50\n")
MD5 = hashlib.md5(NP).hexdigest()


def _exp(acc):
    an = {"@id": "/analyses/ENCAN1/", "status": "released", "assembly": "GRCh38",
          "date_created": "2020"}
    old = {"@id": "/analyses/ENCAN0/", "status": "archived", "assembly": "GRCh38",
           "date_created": "2022"}
    f = lambda a, ot, pd, ans, md5=MD5: {"accession": a, "file_type": "bed narrowPeak",
                                         "assembly": "GRCh38", "status": "released",
                                         "output_type": ot, "preferred_default": pd,
                                         "analyses": [ans], "md5sum": md5,
                                         "href": f"/files/{a}/@@download/{a}.bed.gz",
                                         "date_created": "2020"}
    return {"accession": acc, "default_analysis": "/analyses/ENCAN1/", "analyses": [an, old],
            "files": [f(f"{acc}X", "replicated peaks", True, old),          # archived analysis
                      f(f"{acc}P", "pseudoreplicated peaks", None, an),
                      f(f"{acc}R", "replicated peaks", True, an),           # the pick
                      {"accession": "BAM", "file_type": "bam", "assembly": "GRCh38",
                       "status": "released"}]}


class Portal:
    def __init__(self, ledger_path=None):
        self.calls, self.ledger_path = [], ledger_path

    def __call__(self, url):
        n_ledger = (len(self.ledger_path.read_text().splitlines()) - 1
                    if self.ledger_path and self.ledger_path.is_file() else 0)
        self.calls.append((url, n_ledger))
        if "/experiments/" in url:
            acc = url.split("/experiments/")[1].split("/")[0]
            return json.dumps(_exp(acc)).encode()
        return NP


def _setup(tmp_path):
    led = tmp_path / "ledger.tsv"
    cols = ["eic_track", "role", "cell", "assay", "assay_class", "experiment"]
    with open(led, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        w.writerow(["C01M01", "val", "a", "H3K4me3", "narrow_histone", "ENCSRVAL"])
        w.writerow(["C02M01", "blind", "b", "H3K4me3", "narrow_histone", "ENCSRBLD"])
        w.writerow(["C02CTL", "blind", "b", "chipseq-control", "control", "ENCSRCTL"])
        w.writerow(["C03M01", "train", "c", "H3K4me3", "narrow_histone", "ENCSRTRN"])
    ev = tmp_path / "eval" / "evalset_blind.json"
    ev.parent.mkdir()
    ev.write_text(json.dumps({"tables_sha256": "abc", "regions": [["chr1", 0]]}))
    return led, ev


def _rows(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_fetch_val_picks_rule_and_checks_md5(tmp_path):
    led, _ = _setup(tmp_path)
    out = tmp_path / "peaks"
    assert P.main(["fetch", "--ledger", str(led), "--roles", "train,val", "--out", str(out)],
                  opener=Portal()) == 0
    rows = {r["eic_track"]: r for r in _rows(out / "peak_files.tsv")}
    assert set(rows) == {"C01M01", "C03M01"}
    r = rows["C01M01"]
    assert r["file"] == "ENCSRVALR" and r["md5"] == MD5 and r["n_peaks"] == "2"
    assert r["rule"].startswith("portal:default_analysis;preferred_default=yes")
    assert (out / "C01M01.narrowPeak.gz").read_bytes() == NP
    bad = lambda url: (json.dumps(_exp("ENCSRVAL")).encode() if "/experiments/" in url
                       else b"corrupt")
    out2 = tmp_path / "peaks2"
    assert P.main(["fetch", "--ledger", str(led), "--roles", "val", "--out", str(out2)],
                  opener=bad) == 3
    r2 = _rows(out2 / "peak_files.tsv")[0]
    assert r2["file"] == "" and r2["rule"].startswith("failed: download")
    assert not (out2 / "C01M01.narrowPeak.gz").exists()


def test_roles_required_and_blind_refused_without_flag(tmp_path):
    led, ev = _setup(tmp_path)
    out = tmp_path / "peaks"
    with pytest.raises(SystemExit):
        P.main(["fetch", "--ledger", str(led), "--out", str(out)], opener=Portal())
    portal = Portal()
    assert P.main(["fetch", "--ledger", str(led), "--roles", "val,blind", "--out", str(out)],
                  opener=portal) == 2
    assert P.main(["fetch", "--ledger", str(led), "--roles", "blind", "--allow-blind",
                   "--out", str(out)], opener=portal) == 2          # no --evalset
    assert portal.calls == [] and not (ev.parent / "BLIND_PASSES.tsv").exists()


def test_blind_fetch_writes_two_ledger_rows_and_refuses_a_second(tmp_path):
    led, ev = _setup(tmp_path)
    out = tmp_path / "peaks"
    ledger = ev.parent / "BLIND_PASSES.tsv"
    portal = Portal(ledger)
    args = ["fetch", "--ledger", str(led), "--roles", "val,blind", "--out", str(out),
            "--allow-blind", "--evalset", str(ev)]
    assert P.main(args, opener=portal) == 0
    rows = _rows(ledger)
    assert [r["status"] for r in rows] == ["started", "fetched"]
    assert all(r["run"] == f"peaks:{out.resolve()}" and r["tables_sha256"] == "abc" for r in rows)
    blind_calls = [n for url, n in portal.calls if "ENCSRBLD" in url]
    assert blind_calls and all(n == 1 for n in blind_calls)   # every blind read after `started`
    assert all(n == 0 for url, n in portal.calls if "ENCSRVAL" in url)
    assert not any("ENCSRCTL" in url for url, _ in portal.calls)
    assert {r["eic_track"] for r in _rows(out / "peak_files.tsv")} == {"C01M01", "C02M01"}

    portal2 = Portal()
    assert P.main(args, opener=portal2) == 2
    assert portal2.calls == [] and len(_rows(ledger)) == 2
    assert P.main(args + ["--repeat-blind-reason", "portal outage"], opener=Portal()) == 0
    rows = _rows(ledger)
    assert len(rows) == 4 and rows[-1]["repeat_reason"] == "portal outage"
