"""t392 chunk C5: the recount plan (`tools/t392/build_plan.py`) and the run checks (`tools/t392/check.py`).

Synthetic inputs only. `check.hpc_run` is replaced: the remote command runs in a local bash on a tmp tree,
so the read-only remote programs are exercised and no network is used.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

GIB = 1024 ** 3


def _bp():
    from t392 import build_plan
    return build_plan


def _ck():
    from t392 import check
    return check


# --------------------------------------------------------------------------- build_plan fixtures

def _index_line(member: str, size: int) -> str:
    return f"-rw-r----- mforooz/def-maxwl {size} 2026-11-11 16:35 {member}"


@pytest.fixture
def world(tmp_path):
    """Two tars and a testbed; 7 measurements, one with 2 lines per fragment and 2 files per half."""
    tables, meas, index, testbed = (tmp_path / n for n in ("tables", "meas", "index", "testbed"))
    for d in (tables, meas, index, testbed):
        d.mkdir()
    specs = {  # id: (source, [(half, name, GiB)], lines_per_fragment)
        "A.base": ("onion-work__000", [("pr1", "A/base/pr1", 1.0), ("pr2", "A/base/pr2", 1.0)], 1),
        "B.base": ("onion-work__000", [("pr1", "B/base/pr1", 0.5), ("pr2", "B/base/pr2", 0.5)], 1),
        "C.base": ("onion-work__000", [("pr1", "C/base/pr1", 0.4), ("pr2", "C/base/pr2", 0.4)], 2),
        "D.base": ("onion-work__000", [("pr1", "D/base/pr1", 0.3), ("pr2", "D/base/pr2", 0.3)], 1),
        "E.pe": ("onion-work__001", [("pr1", "E/pe/pr1", 0.2), ("pr1", "E/pe2/pr1", 0.1),
                                      ("pr2", "E/pe/pr2", 0.2), ("pr2", "E/pe2/pr2", 0.1)], 2),
        "T1.testbed": ("testbed", [("pr1", "X__base/f.pr1", 0.01), ("pr2", "X__base/f.pr2", 0.01)], 1),
        "T2.testbed": ("testbed", [("pr1", "Y__depth/g.pr1", 0.02), ("pr2", "Y__depth/g.pr2", 0.02)], 1),
    }
    lines: dict[str, list[str]] = {"onion-work__000": [], "onion-work__001": []}
    for tar in lines:   # a directory line and a non-read member, which the plan ignores
        lines[tar].append("drwxr-x--- mforooz/def-maxwl 0 2026-11-11 16:38 onion/work/X/")
        lines[tar].append(_index_line("onion/work/X/base/rep1.tagAlign.gz", 10 * GIB))
    for mid, (src, files, lpf) in specs.items():
        halves: dict[str, dict] = {"pr1": {"tagalign": []}, "pr2": {"tagalign": []}}
        for half, name, gib in files:
            if src == "testbed":
                p = testbed / f"{name}.tagAlign.gz"
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "wb") as fh:
                    fh.truncate(int(gib * GIB))
                full = f"/project/def-maxwl/mforooz/ONION/testbed/readsets/{name}.tagAlign.gz"
            else:
                member = f"onion/work/{name}.tagAlign.gz"
                lines[src].append(_index_line(member, int(gib * GIB)))
                full = f"/scratch/mforooz/{member}"
            halves[half]["tagalign"].append(full)
        (meas / mid).mkdir()
        (meas / mid / "provenance.json").write_text(json.dumps(
            {"measurement_id": mid, "lines_per_fragment": lpf, "halves": halves}))
    for tar, ls in lines.items():
        (index / f"{tar}.txt").write_text("\n".join(ls) + "\n")
    with open(tables / "measurements.tsv", "w") as fh:
        fh.write("measurement_id\tassay\n" + "".join(f"{m}\tH3K4me3\n" for m in specs))
    return {"tables": tables, "meas": meas, "index": index, "testbed": testbed,
            "out": tmp_path / "plans", "specs": specs}


def _build(w, gib=2.6):
    rc = _bp().main(["--tables", str(w["tables"]), "--measurements", str(w["meas"]),
                     "--index", str(w["index"]), "--testbed-root", str(w["testbed"]),
                     "--gib-per-task", str(gib), "--out", str(w["out"])])
    assert rc == 0
    plans = {}
    for f in sorted(w["out"].glob("plan_*.tsv")):
        with open(f, newline="") as fh:
            plans[f.name] = list(csv.DictReader(fh, delimiter="\t"))
    return plans


# --------------------------------------------------------------------------- build_plan

def test_one_plan_file_per_source_with_the_pinned_columns(world):
    plans = _build(world)
    assert set(plans) == {"plan_onion-work__000.tsv", "plan_onion-work__001.tsv", "plan_testbed.tsv"}
    for name, rows in plans.items():
        with open(world["out"] / name) as fh:
            assert fh.readline().rstrip("\n").split("\t") == list(_bp().COLUMNS)
        assert {r["source"] for r in rows} == {name[len("plan_"):-len(".tsv")]}
    ids = [r["measurement_id"] for rows in plans.values() for r in rows]
    assert sorted(ids) == sorted(world["specs"])            # every measurement once


def test_rows_carry_size_file_count_and_lines_per_fragment(world):
    rows = {r["measurement_id"]: r for rs in _build(world).values() for r in rs}
    for mid, (src, files, lpf) in world["specs"].items():
        assert rows[mid]["source"] == src
        assert float(rows[mid]["read_gib"]) == pytest.approx(sum(g for _, _, g in files), abs=1e-5)
        assert int(rows[mid]["n_tagaligns"]) == len(files)
        assert int(rows[mid]["lines_per_fragment"]) == lpf


def test_tasks_pack_inside_a_source_to_near_equal_gib(world):
    plans = _build(world, gib=1.5)
    rows = plans["plan_onion-work__000.tsv"]                # 2.0 + 1.0 + 0.8 + 0.6 = 4.4 GiB
    loads: dict[str, float] = {}
    for r in rows:
        loads[r["task"]] = loads.get(r["task"], 0.0) + float(r["read_gib"])
    assert sorted(loads) == ["1", "2", "3"]                   # ceil(4.4 / 1.5), numbered from 1
    assert max(loads.values()) - min(loads.values()) <= 2.0 + 1e-9   # no task above the largest item
    assert sum(loads.values()) == pytest.approx(4.4, abs=1e-5)
    for name in ("plan_onion-work__001.tsv", "plan_testbed.tsv"):
        assert {r["task"] for r in plans[name]} == {"1"}      # task numbers restart per source


def test_a_measurement_in_no_source_raises(world):
    p = world["meas"] / "B.base" / "provenance.json"
    d = json.loads(p.read_text())
    d["halves"]["pr2"]["tagalign"] = ["/scratch/mforooz/onion/work/missing/pr2.tagAlign.gz"]
    p.write_text(json.dumps(d))
    with pytest.raises(LookupError):
        _build(world)


def test_members_prints_only_the_plans_pr1_pr2_members(world, capsys):
    _build(world)
    capsys.readouterr()
    bp = _bp()
    assert bp.main(["members", "--plan", str(world["out"] / "plan_onion-work__001.tsv"),
                    "--measurements", str(world["meas"])]) == 0
    got = capsys.readouterr().out.split()
    assert got == sorted(["onion/work/E/pe/pr1.tagAlign.gz", "onion/work/E/pe2/pr1.tagAlign.gz",
                          "onion/work/E/pe/pr2.tagAlign.gz", "onion/work/E/pe2/pr2.tagAlign.gz"])
    assert bp.main(["members", "--plan", str(world["out"] / "plan_onion-work__000.tsv"),
                    "--measurements", str(world["meas"])]) == 0
    got = capsys.readouterr().out.split()
    assert len(got) == 8 and all(m.endswith(("/pr1.tagAlign.gz", "/pr2.tagAlign.gz")) for m in got)
    assert "onion/work/X/base/rep1.tagAlign.gz" not in got
    assert bp.main(["members", "--plan", str(world["out"] / "plan_testbed.tsv"),
                    "--measurements", str(world["meas"])]) == 0
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- check: the hpc mock

@pytest.fixture
def local_hpc(monkeypatch):
    """`hpc run <cluster> '<cmd>'` runs `<cmd>` in a local bash; records every call."""
    calls = []
    ck = _ck()

    def fake(cluster, cmd):
        calls.append((cluster, cmd))
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    monkeypatch.setattr(ck, "hpc_run", fake)
    return calls


def _gate(ref, n=0):
    return {"reference": f"/scratch/mforooz/onion/measurements/m/{ref}", "reference_sha256": "ab",
            "n_bins": 100, "n_differing": n, "result": "pass" if n == 0 else "fail"}


def _measurement(store: Path, mid: str, differing=(0, 0, 0), files=("starts.h5", "starts_pr1.h5",
                                                                    "starts_pr2.h5")):
    d = store / mid
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_bytes(b"h5")
    prov = {"measurement_id": mid, "wall_s": 12.5, "peak_rss_kb": 2048,
            "files": {f: {"sha256": "x", "gate": _gate(r, n), "conversion_floor_nmse": 0.01 * (i + 1)}
                      for i, (f, r, n) in enumerate(zip(("starts.h5", "starts_pr1.h5", "starts_pr2.h5"),
                                                        ("counts.h5", "pr1.h5", "pr2.h5"), differing))}}
    (d / "provenance.json").write_text(json.dumps(prov))


def _tables(tmp_path, ids):
    t = tmp_path / "tables"
    t.mkdir()
    (t / "measurements.tsv").write_text("measurement_id\tassay\n" + "".join(f"{m}\tx\n" for m in ids))
    return t


# --------------------------------------------------------------------------- check probe

def test_probe_passes_and_prints_gate_cost_and_floor(tmp_path, local_hpc, capsys):
    store = tmp_path / "store"
    for m in ("P1.base", "P2.base"):
        _measurement(store, m)
    assert _ck().main(["probe", "--cluster", "nibi", "--store", str(store)]) == 0
    out = capsys.readouterr().out
    assert out.count("differing bins: 0") == 6
    assert "wall_s 12.5" in out and "peak_rss_kb 2048" in out
    assert "conversion_floor_nmse 0.01" in out and "conversion_floor_nmse 0.03" in out
    assert [c[0] for c in local_hpc] == ["nibi"]


def test_probe_fails_on_one_differing_bin(tmp_path, local_hpc, capsys):
    store = tmp_path / "store"
    _measurement(store, "P1.base")
    _measurement(store, "P2.base", differing=(0, 1, 0))
    assert _ck().main(["probe", "--cluster", "nibi", "--store", str(store)]) == 1
    assert "differing bins: 1" in capsys.readouterr().out


def test_probe_fails_on_an_empty_store(tmp_path, local_hpc):
    assert _ck().main(["probe", "--cluster", "nibi", "--store", str(tmp_path / "none")]) == 1


# --------------------------------------------------------------------------- check store

def test_store_passes_when_every_measurement_is_complete(tmp_path, local_hpc, capsys):
    ids = ["M1.base", "M2.depth", "M3.mapq0"]
    store = tmp_path / "store"
    for m in ids:
        _measurement(store, m)
    rc = _ck().main(["store", "--cluster", "fir", "--store", str(store),
                     "--tables", str(_tables(tmp_path, ids))])
    out = capsys.readouterr().out
    assert rc == 0
    assert "measurements 3 of 3" in out and "files 9 of 9" in out and "differing bins: 0" in out


def test_store_counts_missing_files_measurements_and_differing_bins(tmp_path, local_hpc, capsys):
    ids = ["M1.base", "M2.depth", "M3.mapq0", "M4.absent"]
    store = tmp_path / "store"
    _measurement(store, "M1.base")
    _measurement(store, "M2.depth", files=("starts.h5", "starts_pr1.h5"))
    _measurement(store, "M3.mapq0", differing=(2, 0, 3))
    rc = _ck().main(["store", "--cluster", "fir", "--store", str(store),
                     "--tables", str(_tables(tmp_path, ids))])
    out = capsys.readouterr().out
    assert rc == 1
    assert "measurements 1 of 4" in out and "files 8 of 12" in out and "differing bins: 5" in out


# --------------------------------------------------------------------------- check train, scores

def _run(runs: Path, seed, windows=80000, params=993615, done=True, sha="7a1a0325d00ddff3" + "0" * 48):
    d = runs / f"seed{seed}"
    d.mkdir(parents=True)
    (d / "ckpt.pt").write_bytes(b"x")
    if done:
        (d / "DONE").write_text("80000")
    (d / "run.json").write_text(json.dumps({"done": done, "windows_seen": windows, "param_count": params,
                                            "tables_sha256": sha, "device": "cpu"}))
    return d


def test_train_passes_on_three_finished_seeds(tmp_path, local_hpc, capsys):
    runs = tmp_path / "arm"
    for s in (0, 1, 2):
        _run(runs, s)
    assert _ck().main(["train", "--cluster", "fir", "--runs", str(runs)]) == 0
    out = capsys.readouterr().out
    assert "seeds done: 0 1 2" in out and out.count("windows_seen 80000") == 3


@pytest.mark.parametrize("bad", [{"windows": 79999}, {"params": 1_030_000}, {"done": False},
                                 {"sha": "a068a8674aee4252" + "0" * 48}])
def test_train_fails_on_one_bad_seed(tmp_path, local_hpc, capsys, bad):
    runs = tmp_path / "arm"
    _run(runs, 0)
    _run(runs, 1, **bad)
    _run(runs, 2)
    assert _ck().main(["train", "--cluster", "fir", "--runs", str(runs)]) == 1


def test_train_fails_on_a_missing_seed(tmp_path, local_hpc, capsys):
    runs = tmp_path / "arm"
    _run(runs, 0)
    _run(runs, 2)
    assert _ck().main(["train", "--cluster", "fir", "--runs", str(runs)]) == 1
    assert "seeds done: 0 2" in capsys.readouterr().out


def test_scores_counts_scored_seeds(tmp_path, local_hpc, capsys):
    runs = tmp_path / "arm"
    for s in (0, 1, 2):
        d = _run(runs, s) / "p16_sv"
        d.mkdir()
        (d / "scores.json").write_text("{}")
        (d / "SCORED").write_text("t")
    assert _ck().main(["scores", "--cluster", "fir", "--runs", str(runs), "--sub", "p16_sv"]) == 0
    assert "SCORED 3 of 3" in capsys.readouterr().out
    (runs / "seed1" / "p16_sv" / "SCORED").unlink()
    assert _ck().main(["scores", "--cluster", "fir", "--runs", str(runs), "--sub", "p16_sv"]) == 1
    assert "SCORED 2 of 3" in capsys.readouterr().out


# --------------------------------------------------------------------------- read-only, tunnel, spend

def test_checks_never_write_on_the_cluster(tmp_path, local_hpc):
    runs = tmp_path / "arm"
    for s in (0, 1, 2):
        _run(runs, s)
    _measurement(tmp_path / "store", "M1.base")
    before = sorted((p, p.stat().st_mtime_ns) for p in tmp_path.rglob("*"))
    ck = _ck()
    ck.main(["train", "--cluster", "fir", "--runs", str(runs)])
    ck.main(["scores", "--cluster", "fir", "--runs", str(runs), "--sub", "p16_sv"])
    ck.main(["probe", "--cluster", "nibi", "--store", str(tmp_path / "store")])
    assert sorted((p, p.stat().st_mtime_ns) for p in tmp_path.rglob("*")) == before
    for _, cmd in local_hpc:
        assert ">" not in cmd.replace("2>/dev/null", "") and "rm " not in cmd


def test_tunnel_down_exits_69_and_says_hpc_up(monkeypatch, capsys):
    ck = _ck()
    monkeypatch.setattr(ck, "hpc_run", lambda c, cmd: subprocess.CompletedProcess(["hpc"], 69, "", "x"))
    assert ck.main(["train", "--cluster", "fir", "--runs", "/nowhere"]) == 69
    assert "run `hpc up fir`" in capsys.readouterr().err
    assert ck.main(["probe", "--cluster", "nibi", "--store", "/nowhere"]) == 69
    assert "run `hpc up nibi`" in capsys.readouterr().err


def test_hpc_run_is_the_real_command(monkeypatch):
    ck = _ck()
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(ck.subprocess, "run", fake_run)
    ck.hpc_run("fir", "echo hi")
    assert seen["argv"] == ["hpc", "run", "fir", "echo hi"]


def test_spend_sums_cputime_over_files(tmp_path, capsys):
    cols = "job_id\tname\tcluster\tstate\talloc_cpus\tcputime_raw_s\n"
    a, b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    a.write_text(cols + "1\trs392_x\tfir\tCOMPLETED\t2\t7200\n2\trs392_y\tfir\tCOMPLETED\t2\t3600\n")
    b.write_text(cols + "3\trs392_z\tnibi\tCOMPLETED\t1\t1800\n")
    assert _ck().main(["spend", str(a), str(b)]) == 0
    assert "total\t3.50 CPU-h" in capsys.readouterr().out
    b.write_text(cols + "3\trs392_z\tnibi\tRUNNING\t\t\n")
    assert _ck().main(["spend", str(a), str(b)]) == 1
    assert "total\t3.00 CPU-h" in capsys.readouterr().out
