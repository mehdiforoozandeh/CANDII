"""XE: external EIC entries scored in the onion signal space (tools/onion_eval/external_signal.py).

Synthetic tables with three blind H3K4me3 targets that carry pipeline signal tracks; each entry is a
bigWig written by the minimal writer below. Known answers: an entry equal to the truth has r_spec 1,
level 0 and an error budget of 0; its noise floor is the one of the target's own halves."""
from __future__ import annotations

import csv
import json
import struct
import sys
import zlib
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import eic_score as ES  # noqa: E402
from onion_eval import evalset as E  # noqa: E402
from onion_eval import external_signal as X  # noqa: E402
from onion_eval import metrics as Mx  # noqa: E402
from onion_eval import score as S  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"))
BLIND = ((6, "H3K4me3"), (7, "H3K4me3"), (8, "H3K4me3"))
RB = 256
CHROM_BINS = {"chr1": 2048, "chr2": 1024}


# ---- a minimal bigWig v4 writer (as tests/test_onion_eval_eic_score.py) ---------------------
def _write_bigwig(path, chroms, blocks):
    key = max(len(n) for n, _ in chroms)
    ctree = struct.pack("<IIIIQQ", 0x78CA8C91, len(chroms), key, 8, len(chroms), 0)
    ctree += struct.pack("<BBH", 1, 0, len(chroms))
    for i, (n, s) in enumerate(chroms):
        ctree += n.encode().ljust(key, b"\x00") + struct.pack("<II", i, s)
    ctree_off = 64
    data_off = ctree_off + len(ctree)
    data = struct.pack("<Q", len(blocks))
    leaves = []
    for cid, sec in blocks:
        comp = zlib.compress(sec)
        leaves.append((cid, data_off + len(data), len(comp)))
        data += comp
    index_off = data_off + len(data)
    index = struct.pack("<IIQIIIIQII", 0x2468ACE0, len(blocks), len(blocks), 0, 0, 0, 0, 0, 1, 0)
    index += struct.pack("<BBH", 1, 0, len(leaves))
    for cid, off, size in leaves:
        index += struct.pack("<IIIIQQ", cid, 0, cid, 1 << 30, off, size)
    head = struct.pack("<IHHQQQHHQQIQ", 0x888FFC26, 4, 0, ctree_off, data_off, index_off,
                       0, 0, 0, 0, 32768, 0)
    Path(path).write_bytes(head + ctree + data + index)


def _bedgraph(cid, items):
    body = b"".join(struct.pack("<IIf", s, e, v) for s, e, v in items)
    return struct.pack("<IIIIIBBH", cid, items[0][0], items[-1][1], 0, 0, 1, 0, len(items)) + body


def write_track(path, per_chrom: dict):
    """One bedGraph item per 25 bp bin, value = the bin's value (float32)."""
    chroms = [(c, len(v) * ES.WINDOW) for c, v in per_chrom.items()]
    blocks = []
    for cid, v in enumerate(per_chrom.values()):
        items = [(i * 25, (i + 1) * 25, float(x)) for i, x in enumerate(np.asarray(v, np.float32))]
        for a in range(0, len(items), 4000):
            blocks.append((cid, _bedgraph(cid, items[a:a + 4000])))
    _write_bigwig(path, chroms, blocks)


def tsv(p):
    return list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))


# ---- fixture -------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("xe")
    t = make_synthetic_tables(root, n_cells=9, val_tracks=VAL, blind_tracks=BLIND,
                              chrom_bins=CHROM_BINS)
    rows = tsv(t / "measurements.tsv")
    # each blind target has an observed control: the signal route of the training average needs a
    # lambda (score.py; a ChIP target with no control is undefined in r_spec, Log SC OPEN item)
    ctl = {r["cell"]: r["measurement_id"] for r in rows
           if r["assay"] == "chipseq-control" and r["arm_knob"] == ""}
    for r in rows:
        if r["assay"] == "H3K4me3" and r["in_eic"] == "blind" and r["arm_knob"] == "":
            r["control_measurement_id"] = ctl[r["cell"]]
    with open(t / "measurements.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    targets = {}
    for r in rows:
        if r["assay"] == "H3K4me3" and r["in_eic"] in ("val", "blind") and r["arm_knob"] == "":
            d = Path(r["counts_path"][3:]).parent
            for nm, src in (("signal", "counts"), ("pr1_signal", "pr1"), ("pr2_signal", "pr2")):
                with h5py.File(d / f"{src}.h5", "r") as fi, h5py.File(d / f"{nm}.h5", "w") as fo:
                    for c in fi:
                        fo.create_dataset(c, data=np.asarray(fi[c][:], np.float32) * 0.7)
            if r["in_eic"] == "blind":
                targets[r["measurement_id"]] = r
    ev_path = root / "eval" / "synth" / "evalset_blind.json"
    ev = E.build(t, ev_path, include_blind=True, n_regions=3, n_fit_regions=1, region_bins=RB,
                 seed=0)
    ev["imputation_targets"] = [x for x in ev["imputation_targets"] if x["assay"] == "H3K4me3"
                                and x["regime"] == "leave_one_assay_out"]
    ev["correction_pairs"], ev["replicate_pairs"], ev["control_targets"] = [], [], []
    ev["scorable"]["imputation"] = ["H3K4me3"]
    ev_path.write_text(json.dumps(ev))
    blind_ids = sorted({x["target"] for x in ev["imputation_targets"] if x["split"] == "blind"})
    assert len(blind_ids) == 3
    return {"root": root, "tables": t, "evalset": ev_path, "ev": ev, "targets": targets,
            "blind": blind_ids}


def truth_signal(row) -> dict:
    d = Path(row["counts_path"][3:]).parent
    with h5py.File(d / "signal.h5", "r") as f:
        return {c: np.asarray(f[c][:], np.float32) for c in CHROM_BINS}


def entry_dir(root, env, name, transform=lambda row, sig: sig, skip=()):
    d = root / name
    d.mkdir()
    for mid, row in env["targets"].items():
        if mid in skip:
            continue
        write_track(d / f"{row['eic_track']}.bigwig", transform(row, truth_signal(row)))
    return d


def run_score(d, out, env, team=0, extra=()):
    return X.main(["score", "--entry-dir", str(d), "--entry", str(team), "--evalset",
                   str(env["evalset"]), "--allow-blind", "--out", str(out), "--n-boot", "10",
                   *extra])


# ---- the region reader ---------------------------------------------------------------------
def test_read_regions_equals_bin_bigwig_slices(tmp_path):
    rng = np.random.default_rng(1)
    clen = 10_007                                   # not a multiple of 25; 401 bins
    items, pos = [], 3
    while pos < clen - 40:                          # irregular items with gaps
        ln = int(rng.integers(1, 60))
        items.append((pos, pos + ln, float(rng.normal(2, 1.5))))   # some values negative
        pos += ln + int(rng.integers(0, 30))
    p = tmp_path / "r.bigwig"
    _write_bigwig(p, [("chr1", clen), ("chr2", 1_700)],
                  [(0, _bedgraph(0, items[:200])), (0, _bedgraph(0, items[200:])),
                   (1, _bedgraph(1, [(0, 100, 1.0), (130, 140, 5.0)]))])
    full = ES.bin_bigwig(p, ["chr1", "chr2"])
    regions = [("chr1", 0), ("chr1", 337), ("chr2", 0), ("chr1", 100)]
    got = X.read_regions(p, regions, 64)
    for (c, s), y in zip(regions, got):
        assert y.dtype == np.float32
        assert np.array_equal(y.view(np.uint32), full[c][s:s + 64].view(np.uint32)), (c, s)
    # the bin of the last item of chr1 is in region chr1:337 (bins 337..400)
    last = items[-1][1] // 25
    assert 337 <= last < 401
    with pytest.raises(ValueError, match="runs past"):
        X.read_regions(p, [("chr1", 380)], 64)


# ---- scoring -------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def perfect(env, tmp_path_factory):
    root = tmp_path_factory.mktemp("perfect")
    d = entry_dir(root, env, "e0")
    out = root / "e0" / "blind"
    assert run_score(d, out, env, team=0) == 0
    return {"dir": d, "out": out, "s": json.loads((out / "scores.json").read_text())}


def test_perfect_entry_known_answer(env, perfect):
    s = perfect["s"]
    assert s["predictor"] == S.EXTERNAL and s["space"] == "signal_arcsinh"
    assert s["entry"]["team_id"] == 0 and s["entry"]["name"] == "Avocado"
    assert s["split"] == "blind" and s["n_targets"] == 3 and s["targets_without_entry_track"] == []
    # only the signal space is kept
    assert set(s["specific_variance_recovered_above_noise"]) == {"signal_arcsinh"}
    assert set(s["r_spec"]) == {"signal_arcsinh"} and set(s["error_budget"]) == {"signal_arcsinh"}
    for k in S.Scorer.COUNT_SPACE_KEYS + ("selection", "copy_leak", "accessibility_only"):
        assert k not in s
    h1 = s["specific_variance_recovered_above_noise"]["signal_arcsinh"]
    assert h1["macro"] is not None
    assert s["r_spec"]["signal_arcsinh"]["macro"] == pytest.approx(1.0, abs=1e-9)
    regions = [(c, int(st)) for c, st in env["ev"]["regions"]]
    for mid, v in h1["per_target"].items():
        assert v["lambda_source"] == "external_entry"
        assert v["level"] == pytest.approx(0.0, abs=1e-9)
        # noise floor = the target's own halves in signal_arcsinh (known answer)
        d = Path(env["targets"][mid]["counts_path"][3:]).parent
        rd = {nm: Mx.to_space(X._read_h5_regions(d / f"{nm}.h5", regions, RB), "signal_arcsinh")
              for nm in ("signal", "pr1_signal", "pr2_signal")}
        assert v["N"] == pytest.approx(float(Mx.noise_floor(rd["pr1_signal"], rd["pr2_signal"],
                                                            rd["signal"], 2.0)), rel=1e-9)
    bud = s["error_budget"]["signal_arcsinh"]
    for k in ("shared", "specific", "level"):
        assert bud[k] == pytest.approx(0.0, abs=1e-9), k
    assert bud["per_assay"]["H3K4me3"]["n_targets"] == 3
    assert all(v["negative_set_to_0"] == 0 for v in s["entry_values"].values())
    # the full score.py output is kept beside it, and the blind pass is in the ledger
    full = json.loads((perfect["out"] / "scores_full.json").read_text())
    assert full["count_space_reason"] == S.EXTERNAL_REASON
    assert (perfect["out"] / S.SCORED).is_file()
    led = S.ledger_rows(env["evalset"].parent / S.BLIND_LEDGER)
    assert any(r["out"] == str(perfect["out"].resolve()) and r["status"] == "scored" for r in led)


def test_rerun_skips(env, perfect, capsys):
    assert run_score(perfect["dir"], perfect["out"], env) == 0
    assert "skip" in capsys.readouterr().out


def test_noisy_entry_scores_below_perfect(env, perfect, tmp_path):
    rng = np.random.default_rng(3)

    def noisy(row, sig):
        return {c: np.abs(v + rng.normal(0, 2.0, v.shape)).astype(np.float32) for c, v in sig.items()}
    d = entry_dir(tmp_path, env, "e1", noisy)
    out = tmp_path / "e1" / "blind"
    assert run_score(d, out, env, team=100) == 0
    s = json.loads((out / "scores.json").read_text())
    r = s["r_spec"]["signal_arcsinh"]["macro"]
    assert r is not None and r < 0.99
    h_p = perfect["s"]["specific_variance_recovered_above_noise"]["signal_arcsinh"]["macro"]
    h_n = s["specific_variance_recovered_above_noise"]["signal_arcsinh"]["macro"]
    assert h_n < h_p
    assert s["error_budget"]["signal_arcsinh"]["specific"] > 0.01


def test_negative_values_set_to_zero(env, perfect, tmp_path):
    """Negative bins written where the truth is 0 are set to 0: the scores equal the perfect entry's."""
    regions = [(c, int(st)) for c, st in env["ev"]["regions"]]

    def neg(row, sig):
        out = {c: v.copy() for c, v in sig.items()}
        for c, st in regions:                   # truth-0 bins inside the scored regions
            z = st + np.flatnonzero(out[c][st:st + RB] == 0)[:7]
            out[c][z] = -1.5
        return out
    d = entry_dir(tmp_path, env, "e2", neg)
    out = tmp_path / "e2" / "blind"
    assert run_score(d, out, env, team=3393457) == 0
    s = json.loads((out / "scores.json").read_text())
    assert sum(v["negative_set_to_0"] for v in s["entry_values"].values()) > 0
    assert all(v["min_before"] == -1.5 for v in s["entry_values"].values()
               if v["negative_set_to_0"])
    for key in ("specific_variance_recovered_above_noise", "r_spec"):
        assert s[key]["signal_arcsinh"]["macro"] == perfect["s"][key]["signal_arcsinh"]["macro"]


def test_missing_track_is_left_out(env, tmp_path):
    drop = env["blind"][0]
    d = entry_dir(tmp_path, env, "e3", skip=(drop,))
    out = tmp_path / "e3" / "blind"
    assert run_score(d, out, env, team=3391272) == 0
    s = json.loads((out / "scores.json").read_text())
    assert s["targets_without_entry_track"] == [drop] and s["n_targets"] == 2
    used = Path(s["evalset_used"])
    assert used.parent == env["evalset"].parent and used.name == "evalset_blind_xe3391272.json"
    assert E.is_blind_evalset_name(used.name)
    assert s["evalset_base_sha256"] == S.sha256_file(env["evalset"])
    assert drop not in s["specific_variance_recovered_above_noise"]["signal_arcsinh"]["per_target"]


def test_refusals(env, perfect, tmp_path):
    assert run_score(perfect["dir"], tmp_path / "x" / "blind", env, team=12345) == 2
    assert run_score(perfect["dir"], tmp_path / "x" / "notblind", env) == 2   # blind out name
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_score(empty, tmp_path / "y" / "blind", env) == 2


# ---- the table -----------------------------------------------------------------------------
def test_table_puts_entries_beside_the_model(env, perfect, tmp_path):
    # a scored "model" row: score.py's position average on the same blind eval set
    mdir = tmp_path / "runs" / "pa" / "seed0"
    S.score(env["evalset"], mdir / "blind", baseline="position_average", split="blind",
            allow_blind=True, n_boot=10, log=lambda *a: None)
    out = tmp_path / "t.tsv"
    rows = X.build_table([perfect["out"]], [mdir], out)
    got = tsv(out)
    assert [r["kind"] for r in got] == ["model", "external"]
    assert got[1]["name"] == "Avocado" and float(got[1]["r_spec_signal_arcsinh"]) == pytest.approx(1.0)
    assert got[0]["budget_specific"] == "" and got[1]["budget_specific"] != ""
    assert "h1_H3K4me3" in got[0] and got[0]["n_targets"] == "3"
    assert len(rows) == 2
    # a row on another eval set is refused
    s = json.loads((perfect["out"] / "scores.json").read_text())
    s["evalset_base_sha256"] = "0" * 64
    other = tmp_path / "other"
    other.mkdir()
    (other / "scores.json").write_text(json.dumps(s))
    with pytest.raises(ValueError, match="eval set sha256"):
        X.build_table([perfect["out"], other], [mdir], tmp_path / "t2.tsv")
