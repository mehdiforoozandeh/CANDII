"""Tests for the AG2 fetch tool (onion Amendment 5, chunk AGf). A fake client only; no API call, no alphagenome."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from onion_data import ag2_fetch as F  # noqa: E402

KEY = "SECRETkey-0123456789abcdef"
PROJECT = "projects/987654321"
CHIP = ["H3K4me3 single", "H3K4me3 paired", "H3K27ac single"]
CHIP_MODES = ["observed", "imputed_supported", "observed"]
ACC = ["DNASE single", "DNASE paired"]
ACC_MODES = ["imputed_supported", "observed"]
RB = 64                                     # small regions: 1,600 bp, not on the 128 bp grid


def chip_value(gbin, j):
    return (np.asarray(gbin) % 97).astype(np.float64) + 10.0 * j


def acc_value(pos, j):
    return (np.asarray(pos) % 7).astype(np.float64) + j


class QuotaError(Exception):
    """Looks like a grpc error: code() -> RESOURCE_EXHAUSTED; the message quotes the key and a project."""

    class _Code:
        name = "RESOURCE_EXHAUSTED"

    def __init__(self):
        super().__init__(f"Quota exceeded for {PROJECT} with key {KEY}")

    def code(self):
        return self._Code()


class FakeClient:
    """`permute`: the channel order differs between windows of a call and between calls (fact (d) of the API
    smoke: match by name on every response)."""

    def __init__(self, errors=(), permute=False):
        self.calls, self.errors, self.permute = [], list(errors), permute

    def predict(self, intervals, condition, outputs, workers):
        self.calls.append([tuple(i) for i in intervals])
        if self.errors:
            e = self.errors.pop(0)
            if e is not None:
                raise e
        out = []
        for c, s, e in intervals:
            assert e - s == F.W and s % F.RES == 0
            d = {}
            gb = s // F.RES + np.arange(F.W // F.RES)
            d["chip_histone"] = (np.stack([chip_value(gb, j) for j in range(len(CHIP))], 1),
                                 np.array(CHIP), np.array(CHIP_MODES))
            for o in outputs:
                if o != "chip_histone":
                    pos = s + np.arange(F.W)
                    d[o] = (np.stack([acc_value(pos, j) for j in range(len(ACC))], 1), np.array(ACC),
                            np.array(ACC_MODES))
            if self.permute and (len(out) + len(self.calls)) % 2:
                d = {o: (v[:, ::-1], n[::-1], m[::-1]) for o, (v, n, m) in d.items()}
            out.append({o: d[o] for o in outputs})
        return out


@pytest.fixture
def keyfile(tmp_path, monkeypatch):
    p = tmp_path / "key"
    p.write_text(KEY + "\n")
    os.chmod(p, 0o600)
    monkeypatch.setenv("AG2_KEY_FILE", str(p))
    return p


def _evalset(path, starts=(("chr21", 64 * 400), ("chr21", 64 * 9000), ("chr21", 64 * 20000), ("chr21", 0))):
    ev = {"regions": [list(r) for r in starts], "region_bins": RB, "imputation_targets": []}
    path.write_text(json.dumps(ev))
    return ev


def _conds(path, conds=("EFO:0001", "CL:0002")):
    path.write_text("condition\tn_val_tracks\n" + "".join(f"{c}\t1\n" for c in conds))
    return list(conds)


def _run(tmp_path, client, *extra, outputs="chip_histone", conds=("EFO:0001",), sleeps=None):
    ev, cf, cache = tmp_path / "ev.json", tmp_path / "conds.tsv", tmp_path / "cache"
    if not ev.exists():
        _evalset(ev)
    _conds(cf, conds)
    rec = sleeps if sleeps is not None else []
    rc = F.main(["fetch", "--conditions", str(cf), "--evalset", str(ev), "--outputs", outputs,
                 "--cache", str(cache), "--chunk", "2", *extra],
                client_factory=lambda key, model: client, sleep=rec.append)
    return rc, cache


def _parts(cache, cond="EFO:0001"):
    return sorted(p.name for p in F.cond_dir(cache, "all_folds", cond).iterdir())


# ------------------------------------------------------------------------------------------------ windows

def test_evalset_windows_centred_and_shifted_inward():
    ev = {"regions": [["chr21", 64 * 9000], ["chr21", 0], ["chr21", (F.HG38["chr21"] // 25 // RB - 1) * RB]],
          "region_bins": RB}
    ws = F.evalset_windows(ev)
    w = ws[0]
    assert w["ws"] % F.RES == 0 and w["ws"] <= w["rs"] and w["re"] <= w["we"]
    assert abs((w["ws"] + F.W // 2) - (w["rs"] + w["re"]) // 2) < F.RES          # centred
    assert ws[1]["ws"] == 0                                                        # shifted at the start
    assert ws[2]["we"] <= F.HG38["chr21"] and ws[2]["re"] <= ws[2]["we"]           # shifted at the end
    assert all(x["n_chip"] * F.RES >= x["re"] - x["rs"] + 2 * F.MARGIN_BINS * F.RES for x in ws)


def test_tile_windows_match_max_count():
    # Max's tiling gives 122 + 89 + 96 = 307 windows on chr20 to chr22 (his measured 23,946 calls / 78 = 307;
    # plan B.4's ceil(L / stride) = 310 overcounts)
    ws = F.tile_windows(["chr20", "chr21", "chr22"])
    assert len(ws) == 307
    for c in ("chr20", "chr21", "chr22"):
        cw = [w for w in ws if w["chrom"] == c]
        assert cw[0]["rs"] == 0 and cw[-1]["re"] == F.HG38[c] // F.RES * F.RES
        assert all(a["re"] == b["rs"] for a, b in zip(cw[:-1], cw[1:]))              # bins partition the chrom
        assert all(w["ws"] <= w["rs"] and w["re"] <= w["we"] for w in cw)


# -------------------------------------------------------------------------------------------- fetch rules

def test_kill_between_chunks_loses_no_finished_chunk(tmp_path, keyfile):
    rc, cache = _run(tmp_path, FakeClient())
    assert rc == 0
    full = {p: np.load(F.cond_dir(cache, "all_folds", "EFO:0001") / p)["chip_histone_0"] for p in _parts(cache)}
    import shutil
    shutil.rmtree(cache)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, FakeClient(errors=[None, KeyboardInterrupt()]))
    assert _parts(cache) == ["part_0.npz"]                                          # no tmp file left
    z = np.load(F.cond_dir(cache, "all_folds", "EFO:0001") / "part_0.npz")
    assert np.array_equal(z["chip_histone_0"], full["part_0.npz"], equal_nan=True)
    c2 = FakeClient()
    rc, _ = _run(tmp_path, c2)
    assert rc == 0 and len(c2.calls) == 1 and _parts(cache) == ["part_0.npz", "part_2.npz"]


def test_rerun_calls_only_missing_windows(tmp_path, keyfile):
    c1 = FakeClient()
    rc, cache = _run(tmp_path, c1, conds=("EFO:0001", "CL:0002"))
    assert rc == 0 and sum(len(c) for c in c1.calls) == 8
    os.remove(F.cond_dir(cache, "all_folds", "CL:0002") / "part_2.npz")
    c2 = FakeClient()
    rc, _ = _run(tmp_path, c2, conds=("EFO:0001", "CL:0002"))
    ev = F.evalset_windows(json.loads((tmp_path / "ev.json").read_text()))
    assert rc == 0 and c2.calls == [[(w["chrom"], w["ws"], w["we"]) for w in ev[2:4]]]
    c3 = FakeClient()
    rc, _ = _run(tmp_path, c3, conds=("EFO:0001", "CL:0002"))
    assert rc == 0 and c3.calls == []
    # a new output makes every window missing again for that output only
    c4 = FakeClient()
    rc, _ = _run(tmp_path, c4, outputs="chip_histone,dnase", conds=("EFO:0001",))
    assert rc == 0 and sum(len(c) for c in c4.calls) == 4


def test_three_quota_errors_then_success_waits_three_times(tmp_path, keyfile):
    sleeps = []
    c = FakeClient(errors=[QuotaError(), QuotaError(), QuotaError()])
    rc, cache = _run(tmp_path, c, sleeps=sleeps)
    assert rc == 0 and sleeps == [30, 60, 120]
    rows = list(csv.DictReader(open(cache / "calls.tsv"), delimiter="\t"))
    assert list(rows[0]) == list(F.CALLS_COLUMNS)
    assert [r["status"] for r in rows].count("RESOURCE_EXHAUSTED") == 3 * 2       # 2 windows per chunk
    assert [r["status"] for r in rows].count("OK") == 4


def test_eight_quota_errors_exit_75_with_consistent_cache(tmp_path, keyfile):
    sleeps = []
    c = FakeClient(errors=[None] + [QuotaError()] * 8)
    rc, cache = _run(tmp_path, c, sleeps=sleeps)
    assert rc == 75 and sleeps == [30, 60, 120, 240, 480, 600, 600, 600]
    assert _parts(cache) == ["part_0.npz"]
    windows = F.evalset_windows(json.loads((tmp_path / "ev.json").read_text()))
    assert F.missing_windows(cache, "all_folds", "EFO:0001", windows, ["chip_histone"]) == [2, 3]


def test_non_retryable_error_exits_1(tmp_path, keyfile, capsys):
    class Denied(Exception):
        def code(self):
            return type("C", (), {"name": "PERMISSION_DENIED"})()
    rc, cache = _run(tmp_path, FakeClient(errors=[Denied(f"{PROJECT} {KEY}")]))
    assert rc == 1 and "PERMISSION_DENIED" in capsys.readouterr().err


def test_max_calls_stops_cleanly(tmp_path, keyfile, capsys):
    c = FakeClient()
    rc, cache = _run(tmp_path, c, "--max-calls", "3")
    assert rc == 75 and sum(len(x) for x in c.calls) == 2 and _parts(cache) == ["part_0.npz"]
    assert "max-calls" in capsys.readouterr().out
    ev, cf = tmp_path / "ev.json", tmp_path / "conds.tsv"
    assert F.main(["status", "--conditions", str(cf), "--evalset", str(ev), "--cache", str(cache)]) == 0
    assert "windows: 4, cached: 2, missing: 2" in capsys.readouterr().out


def test_no_missing_window_exits_0_without_reading_key(tmp_path, keyfile):
    rc, cache = _run(tmp_path, FakeClient())
    os.chmod(keyfile, 0o644)                          # the key is not needed when nothing is missing
    rc, _ = _run(tmp_path, FakeClient())
    assert rc == 0


def test_key_never_written_or_printed(tmp_path, keyfile, capsys):
    c = FakeClient(errors=[QuotaError(), QuotaError()])
    rc, cache = _run(tmp_path, c, sleeps=[])
    assert rc == 0
    cap = capsys.readouterr()
    for secret in (KEY, PROJECT):
        assert secret not in cap.out and secret not in cap.err
        for f in cache.rglob("*"):
            if f.is_file():
                assert secret.encode() not in f.read_bytes(), f
                if f.suffix == ".npz":
                    with np.load(f) as z:
                        for k in z.files:
                            if z[k].dtype.kind in "US":
                                assert not any(secret in str(s) for s in z[k].ravel()), (f, k)


@pytest.mark.parametrize("content,mode", [(KEY + "\n", 0o644), (KEY + "\n\n", 0o600), ("a\nb\n", 0o600),
                                          ("\n", 0o600), ("", 0o600)])
def test_bad_key_file_refused(tmp_path, keyfile, capsys, content, mode):
    keyfile.write_text(content)
    os.chmod(keyfile, mode)
    made = []
    ev, cf = tmp_path / "ev.json", tmp_path / "conds.tsv"
    _evalset(ev)
    _conds(cf)
    rc = F.main(["fetch", "--conditions", str(cf), "--evalset", str(ev), "--outputs", "chip_histone",
                 "--cache", str(tmp_path / "cache")], client_factory=lambda k, m: made.append(1), sleep=lambda s: None)
    cap = capsys.readouterr()
    assert rc == 2 and not made and "refused" in cap.err and KEY not in cap.err + cap.out


def test_module_imports_without_alphagenome():
    code = ("import sys; sys.modules['alphagenome'] = None; sys.path.insert(0, %r); "
            "from onion_data import ag2_fetch; "
            "assert sys.modules['alphagenome'] is None; "
            "assert not [m for m in sys.modules if m.startswith('alphagenome.')]; "
            "assert 'torch' not in sys.modules and 'h5py' not in sys.modules and 'candi' not in sys.modules; "
            "print('ok')") % str(_ROOT / "tools")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_real_client_needs_alphagenome(monkeypatch):
    monkeypatch.setitem(sys.modules, "alphagenome", None)
    with pytest.raises(ImportError):
        F._client("x", "all_folds")


# ------------------------------------------------------------------------------------------------- pack

def _resample_bp(values_per_bp, start, n_bins):
    return values_per_bp[start:start + n_bins * 25].reshape(n_bins, 25).mean(1)


def test_resample_mean_hand_values():
    x = np.array([1.0, 3.0, 5.0])                                   # 128 bp bins from 0
    got = F.resample_mean(x, 0, 128, 100, 25, 2)                    # [100,125): 1; [125,150): (3*1 + 22*3)/25
    assert np.allclose(got, [1.0, (3 * 1 + 22 * 3) / 25])
    assert np.allclose(F.resample_mean(np.full(10, 2.5), 0, 128, 7, 25, 40), 2.5)
    assert np.isnan(F.resample_mean(x, 0, 128, 1000, 25, 1)[0])


def test_resample_mean_equals_ag2_space():
    S = pytest.importorskip("onion_eval.ag2_space")
    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, 3.0, 200)
    for args in [(x, 0, 128, 0, 25, 1000), (x, 4096, 128, 4000, 25, 1100), (x[:40], 0, 25, 0, 128, 7)]:
        assert np.allclose(F.resample_mean(*args), S.resample_mean(*args), equal_nan=True)


def _pack_fixture(tmp_path):
    ev = _evalset(tmp_path / "ev.json")
    ev["imputation_targets"] = [
        {"target": "m_k4_val", "regime": "leave_one_assay_out", "split": "val", "assay": "H3K4me3", "cell": "a"},
        {"target": "m_k4_val", "regime": "accessibility_only", "split": "val", "assay": "H3K4me3", "cell": "a"},
        {"target": "m_dn_val", "regime": "leave_one_assay_out", "split": "val", "assay": "DNase-seq", "cell": "a"},
        {"target": "m_k27_val", "regime": "leave_one_assay_out", "split": "val", "assay": "H3K27ac", "cell": "b"},
        {"target": "m_x_val", "regime": "leave_one_assay_out", "split": "val", "assay": "H3K9me3", "cell": "d"},
        {"target": "m_k4_blind", "regime": "leave_one_assay_out", "split": "blind", "assay": "H3K4me3",
         "cell": "c"}]
    (tmp_path / "ev.json").write_text(json.dumps(ev))
    t = tmp_path / "tables"
    t.mkdir()
    (t / "measurements.tsv").write_text(
        "measurement_id\tassay\tcell\teic_track\tin_eic\n"
        "m_k4_val\tH3K4me3\ta\tV01\tval\nm_dn_val\tDNase-seq\ta\tV02\tval\n"
        "m_k27_val\tH3K27ac\tb\tV03\tval\nm_x_val\tH3K9me3\td\tV04\tval\nm_k4_blind\tH3K4me3\tc\tB01\tblind\n")
    (tmp_path / "audit.tsv").write_text(
        "eic_track\trole\tcondition\tstatus\tmatched_channel\totherend_channel\tqueryable\n"
        f"V01\tval\tEFO:0001\tunseen\t{CHIP[0]}\t{CHIP[1]}\t1\n"
        f"V02\tval\tEFO:0001\tunseen\t{ACC[1]}\t{ACC[0]}\t1\n"
        f"V03\tval\tEFO:0001\tobserved_other_read_type\t\t{CHIP[2]}\t1\n"
        f"V04\tval\t\tno_condition\t\t\t0\n"
        f"B01\tblind\tEFO:0001\tobserved_matched\t{CHIP[0]}\t{CHIP[1]}\t1\n")
    return ev


@pytest.mark.parametrize("permute", [False, True])
def test_pack_writes_interface_f(tmp_path, keyfile, permute):
    ev = _pack_fixture(tmp_path)
    rc, cache = _run(tmp_path, FakeClient(permute=permute), outputs="chip_histone,dnase")
    assert rc == 0
    out = tmp_path / "pack"
    rc = F.main(["pack", "--cache", str(cache), "--evalset", str(tmp_path / "ev.json"), "--audit",
                 str(tmp_path / "audit.tsv"), "--tables", str(tmp_path / "tables"), "--split", "val",
                 "--out", str(out)])
    assert rc == 0
    d = out / "leave_one_assay_out"
    meta = json.loads((d / "tracks.json").read_text())
    assert meta["kind"] == "ag2" and meta["split"] == "val" and meta["region_bins"] == RB
    assert meta["n_regions"] == 4 and meta["targets"] == ["m_k4_val", "m_dn_val", "m_k27_val"]
    assert meta["chip_bin_value"] == "sum" and meta["chip_native_divisor"] == 128
    assert meta["regions_sha256"] == F.regions_sha256(ev) and meta["skipped"] == {"status_no_condition": 1}
    assert (d / "EVALUATION_ONLY").read_bytes() == b""
    assert sorted(p.name for p in (out / "accessibility_only").glob("*.npy")) == [
        "m_k4_val.matched.signal.npy", "m_k4_val.native.npy", "m_k4_val.otherend.signal.npy"]
    rows = list(csv.DictReader(open(d / "channels.tsv"), delimiter="\t"))
    assert list(rows[0]) == list(F.CHANNEL_COLUMNS)
    assert rows[0]["api_label_matched"] == "observed" and rows[0]["api_label_otherend"] == "imputed_supported"
    assert (rows[0]["matched_source"], rows[0]["output_type"], rows[0]["native_divisor"]) == (
        "matched", "chip_histone", "128")
    assert (rows[1]["output_type"], rows[1]["native_divisor"], rows[1]["chip_bin_value"]) == ("dnase", "1", "")
    # no channel of the target's read type: the other one is written as matched and labelled so
    assert rows[2]["matched_source"] == "otherend_fallback" and rows[2]["matched_channel"] == CHIP[2]
    k27 = np.load(d / "m_k27_val.matched.signal.npy")
    assert np.array_equal(k27, np.load(d / "m_k27_val.otherend.signal.npy"))
    # second method: expand the fake signal to 1 bp and average 25 bp bins directly
    k4 = np.load(d / "m_k4_val.matched.signal.npy")
    k4o = np.load(d / "m_k4_val.otherend.signal.npy")
    dn = np.load(d / "m_dn_val.matched.signal.npy")
    assert k4.dtype == np.float32 and k4.shape == (4 * RB,) and not np.isnan(k4).any()
    # fact (b): a ChIP-seq API value is the sum over its 128 bp; native = fold change per bp = value / 128
    assert np.allclose(np.load(d / "m_k4_val.native.npy"), k4 / 128.0)
    assert np.array_equal(np.load(d / "m_dn_val.native.npy"), np.load(d / "m_dn_val.matched.signal.npy"))
    for i, (c, sb) in enumerate(ev["regions"]):
        rs = sb * 25
        lo = rs // 128 * 128
        per_bp = np.repeat(chip_value(np.arange(lo // 128, lo // 128 + RB), 0), 128)
        want = _resample_bp(per_bp, rs - lo, RB)
        assert np.allclose(k4[i * RB:(i + 1) * RB], want, atol=1e-3)
        assert np.allclose(k4o[i * RB:(i + 1) * RB], want + 10.0, atol=1e-3)
        assert np.allclose(k27[i * RB:(i + 1) * RB], want + 20.0, atol=1e-3)
        pos = rs + np.arange(RB * 25)
        assert np.allclose(dn[i * RB:(i + 1) * RB], acc_value(pos, 1).reshape(RB, 25).sum(1))
    # the blind split packs the blind target only
    rc = F.main(["pack", "--cache", str(cache), "--evalset", str(tmp_path / "ev.json"), "--audit",
                 str(tmp_path / "audit.tsv"), "--tables", str(tmp_path / "tables"), "--split", "blind",
                 "--out", str(tmp_path / "pack_blind")])
    assert rc == 0
    assert json.loads((tmp_path / "pack_blind/leave_one_assay_out/tracks.json").read_text())["targets"] == [
        "m_k4_blind"]


def test_pack_refuses_incomplete_cache(tmp_path, keyfile, capsys):
    _pack_fixture(tmp_path)
    rc, cache = _run(tmp_path, FakeClient(), "--max-calls", "2", outputs="chip_histone,dnase")
    assert rc == 75
    rc = F.main(["pack", "--cache", str(cache), "--evalset", str(tmp_path / "ev.json"), "--audit",
                 str(tmp_path / "audit.tsv"), "--tables", str(tmp_path / "tables"), "--split", "val",
                 "--out", str(tmp_path / "pack")])
    assert rc == 1 and "not cached" in capsys.readouterr().err


def test_cache_of_other_windows_refused(tmp_path, keyfile, capsys):
    rc, cache = _run(tmp_path, FakeClient())
    assert rc == 0
    _evalset(tmp_path / "ev.json", starts=(("chr21", 64 * 500), ("chr21", 64 * 9000)))
    rc, _ = _run(tmp_path, FakeClient())
    assert rc == 1 and "other coordinates" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------ review retry 1

def test_max_calls_counts_every_retry(tmp_path, keyfile, capsys):
    sleeps = []
    c = FakeClient(errors=[QuotaError()] * 20)                      # always fails
    rc, cache = _run(tmp_path, c, "--max-calls", "7", sleeps=sleeps)
    # chunk of 2: attempts at 2, 4, 6 calls; the 4th attempt would reach 8 > 7
    assert rc == 75 and len(c.calls) == 3 and sleeps == [30, 60, 120]
    assert "max-calls" in capsys.readouterr().out


def test_client_creation_error_hides_key(tmp_path, keyfile, capsys):
    def boom(key, model):
        raise RuntimeError(f"bad key {key} for {PROJECT}")
    ev, cf = tmp_path / "ev.json", tmp_path / "conds.tsv"
    _evalset(ev)
    _conds(cf)
    rc = F.main(["fetch", "--conditions", str(cf), "--evalset", str(ev), "--outputs", "chip_histone",
                 "--cache", str(tmp_path / "cache")], client_factory=boom, sleep=lambda s: None)
    cap = capsys.readouterr()
    assert rc == 1 and "RuntimeError" in cap.err
    for secret in (KEY, PROJECT):
        assert secret not in cap.out + cap.err


def test_client_call_error_with_key_hides_key(tmp_path, keyfile, capsys):
    rc, cache = _run(tmp_path, FakeClient(errors=[ValueError(f"echo {KEY} {PROJECT}")]))
    cap = capsys.readouterr()
    assert rc == 1 and "ValueError" in cap.err
    for secret in (KEY, PROJECT):
        assert secret not in cap.out + cap.err
        assert secret not in (cache / "calls.tsv").read_text()


def test_dry_run_plans_without_key_or_client(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AG2_KEY_FILE", str(tmp_path / "absent"))
    ev, cf = tmp_path / "ev.json", tmp_path / "conds.tsv"
    _evalset(ev)
    _conds(cf, ("EFO:0001", "CL:0002"))
    made = []
    rc = F.main(["fetch", "--dry-run", "--conditions", str(cf), "--evalset", str(ev), "--outputs",
                 "chip_histone", "--cache", str(tmp_path / "cache")],
                client_factory=lambda k, m: made.append(1), sleep=lambda s: None)
    out = capsys.readouterr().out
    assert rc == 0 and not made
    assert "EFO:0001\tmissing windows: 4, planned calls: 4" in out and "total planned calls: 8" in out


def test_region_outside_shifted_window_raises():
    sizes = {"chrT": F.W + 100}
    ev = {"regions": [["chrT", 41880]], "region_bins": RB}          # ends at 1,048,600 > window end 1,048,576
    with pytest.raises(ValueError, match="not inside its window"):
        F.evalset_windows(ev, sizes)


def test_corrupt_part_is_refetched_unless_strict(tmp_path, keyfile, capsys):
    rc, cache = _run(tmp_path, FakeClient())
    assert rc == 0
    bad = F.cond_dir(cache, "all_folds", "EFO:0001") / "part_2.npz"
    bad.write_bytes(b"not a zip file")
    ev, cf = tmp_path / "ev.json", tmp_path / "conds.tsv"
    assert F.main(["status", "--strict", "--conditions", str(cf), "--evalset", str(ev),
                   "--cache", str(cache)]) == 1
    assert "part_2.npz" in capsys.readouterr().err
    c = FakeClient()
    rc, _ = _run(tmp_path, c)
    cap = capsys.readouterr()
    assert rc == 0 and "part_2.npz" in cap.err and sum(len(x) for x in c.calls) == 2
    with np.load(bad) as z:
        assert z["window_index"].tolist() == [2, 3]


def test_missing_imputation_mode_fails():
    with pytest.raises(F.FetchError, match="imputation_mode"):
        F._track("chip_histone", np.zeros((2, 1)), {"name": ["a"]})
    v, n, m = F._track("chip_histone", np.zeros((2, 1)), {"name": ["a"], "imputation_mode": ["observed"]})
    assert list(n) == ["a"] and list(m) == ["observed"]


def test_regions_sha256_equals_ag2_baselines():
    B = pytest.importorskip("onion_eval.ag2_baselines")
    ev = {"regions": [["chr1", 16384], ["chrX", 0]], "region_bins": 16384}
    assert F.regions_sha256(ev) == B.regions_sha256(ev["regions"], ev["region_bins"])
