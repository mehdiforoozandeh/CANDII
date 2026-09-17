"""t112 C10 — `tools/t112/checks.py`: the step-5 pre-use checks and the structural ones beside them.

The depth-law tests build reads, thin them the way the pipeline's subsampler does (a draw of L reads
without replacement), and bin both sides with the store overlap rule. A thinned arm must pass; two
arms of the same size that were NOT made by thinning reads must fail. The thresholds come from the
plan Log and are not restated here as tunables: the tests read them from the module and check them
against the plan's numbers once.

C3's `records.py` is written in parallel, so `structure` runs against a stub validator that honours
C3's pinned CLI (`validate --products D --rows R --expect N` → problem lines, then `OK <n>`).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "t112" / "checks.py"
ARMS = REPO / "tools" / "t112" / "arms.py"
RES = 25


def _load():
    spec = importlib.util.spec_from_file_location("t112_checks", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["t112_checks"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def checks():
    return _load()


# --- synthetic reads and the store overlap rule ---------------------------------------------------

def store_counts(starts: np.ndarray, ends: np.ndarray, length: int) -> np.ndarray:
    """+1 to bins floor(start/25)..floor(end/25) inclusive on a grid of len//25 + 1, truncated."""
    n = length // RES
    d = np.bincount(starts // RES, minlength=n + 2)[: n + 2].astype(np.int64)
    d -= np.bincount(ends // RES + 1, minlength=n + 2)[: n + 2]
    return np.cumsum(d)[:n].astype(np.uint32)


def make_reads(rng, chroms, bins_per_chrom, cover=2.0, read_len=100):
    """Uniform read starts; mean overlap ≈ `cover` reads per bin."""
    length = bins_per_chrom * RES
    n_reads = int(bins_per_chrom * cover / (read_len / RES + 1))
    reads = {}
    for c in chroms:
        s = np.sort(rng.integers(0, length - read_len, n_reads))
        reads[c] = (s, s + read_len)
    return reads, length


def counts_of(reads, keep, length):
    """keep: per chrom a boolean mask over that chrom's reads."""
    return {c: store_counts(s[keep[c]], e[keep[c]], length) for c, (s, e) in reads.items()}


def thin_reads(rng, reads, n_keep):
    """The pipeline subsampler's law: exactly n_keep reads drawn without replacement, genome-wide."""
    chroms = list(reads)
    sizes = [reads[c][0].size for c in chroms]
    pick = np.zeros(sum(sizes), dtype=bool)
    pick[rng.choice(pick.size, n_keep, replace=False)] = True
    out, off = {}, 0
    for c, n in zip(chroms, sizes):
        out[c] = pick[off: off + n]
        off += n
    return out


def first_reads(reads, n_keep):
    """NOT thinning: the first n_keep reads in file (genome) order, as a `head -n` would give."""
    out, left = {}, n_keep
    for c, (s, _) in reads.items():
        take = min(left, s.size)
        m = np.zeros(s.size, dtype=bool)
        m[:take] = True
        out[c] = m
        left -= take
    return out


@pytest.fixture(scope="module")
def genome(checks):
    rng = np.random.default_rng(112)
    reads, length = make_reads(rng, checks.MAIN_CHROMS, bins_per_chrom=100_000)
    n_reads = sum(s.size for s, _ in reads.values())
    base = counts_of(reads, {c: np.ones(s.size, bool) for c, (s, _) in reads.items()}, length)
    return {"reads": reads, "length": length, "n_reads": n_reads, "base": base, "rng": rng}


# --- constants --------------------------------------------------------------------------------------

def test_thresholds_are_the_plan_log_values(checks):
    assert checks.DEPTH_TOTAL_TOL == 0.005
    assert checks.DEPTH_MEAN_TOL == 0.02
    assert (checks.DEPTH_VAR_LO, checks.DEPTH_VAR_HI) == (0.85, 1.15)
    assert checks.DEPTH_K_MAX == 10
    assert checks.DEPTH_MIN_BINS == 10000
    assert checks.BASE_REBUILD_TOL == 1e-6 and checks.BASE_REBUILD_CHROM == "chr21"
    assert len(checks.MAIN_CHROMS) == 23 and checks.MAIN_CHROMS[-1] == "chrX"
    assert checks.P_ONLY_ARMS == ("ratio", "ctlid", "ctldepth", "extsize")


# --- depth_law ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0.5, 0.25])
def test_depth_law_passes_a_per_read_thinned_arm(checks, genome, p):
    n_keep = int(round(p * genome["n_reads"]))
    arm = counts_of(genome["reads"], thin_reads(genome["rng"], genome["reads"], n_keep),
                    genome["length"])
    ok, detail = checks.depth_law(genome["base"], arm, n_keep / genome["n_reads"])
    assert ok, detail
    assert detail["total_pass"] and detail["n_k_tested"] >= 3
    assert all(v["pass"] for v in detail["per_k"].values())
    assert all(v["n_bins"] >= 10000 for v in detail["per_k"].values())


def test_depth_law_fails_first_reads_not_thinned(checks, genome):
    """Same read count, but a prefix of the file: the total and the mean are right (half the genome
    kept whole, half dropped), the Binomial variance is not (it is k²p(1−p), not kp(1−p))."""
    p = 0.5
    n_keep = int(round(p * genome["n_reads"]))
    arm = counts_of(genome["reads"], first_reads(genome["reads"], n_keep), genome["length"])
    ok, detail = checks.depth_law(genome["base"], arm, n_keep / genome["n_reads"])
    assert not ok
    assert detail["total_pass"]                      # it fails on the law, not on the total
    assert not all(v["pass"] for v in detail["per_k"].values())


def test_depth_law_fails_bin_scaled_counts(checks, genome):
    """Bins scaled and rounded: the mean is near k·p but the Binomial variance is gone."""
    p = 0.5
    arm = {c: np.floor(b * p + 0.5).astype(np.uint32) for c, b in genome["base"].items()}
    ok, detail = checks.depth_law(genome["base"], arm, p)
    assert not ok
    assert any(not (0.85 <= v["var_ratio"] <= 1.15) for v in detail["per_k"].values())


def test_depth_law_fails_on_wrong_p_and_on_missing_chrom(checks, genome):
    n_keep = int(round(0.5 * genome["n_reads"]))
    arm = counts_of(genome["reads"], thin_reads(genome["rng"], genome["reads"], n_keep),
                    genome["length"])
    ok, detail = checks.depth_law(genome["base"], arm, 0.49)
    assert not ok and not detail["total_pass"]
    short = dict(arm)
    del short["chr21"]
    ok, detail = checks.depth_law(genome["base"], short, n_keep / genome["n_reads"])
    assert not ok and detail["missing_or_misshapen"] == ["chr21"]


def test_depth_law_fails_when_no_k_has_enough_bins(checks):
    base = {c: np.full(100, 3, np.uint32) for c in checks.MAIN_CHROMS}
    arm = {c: np.full(100, 1, np.uint32) for c in checks.MAIN_CHROMS}
    ok, detail = checks.depth_law(base, arm, 1 / 3)
    assert not ok and detail["n_k_tested"] == 0


def _write_npz(path, arrays):
    import bin25  # sibling of checks.py; on sys.path once checks is loaded
    path.parent.mkdir(parents=True, exist_ok=True)
    bin25.write_npz(path, arrays)


def test_depth_law_through_run_and_summary(checks, genome, tmp_path):
    """p comes from the row's knob_value (reads kept) over the base covariates' depth."""
    cf = tmp_path / "cf"
    track, n = "C19M16", genome["n_reads"]
    _write_npz(cf / "products" / f"{track}__base__base" / "counts25.npz", genome["base"])
    (cf / "products" / f"{track}__base__base" / "covariates.json").write_text(json.dumps({"depth": n}))
    header = "pid\tbiosample\ttrack\tcell\tassay\tarm\tlevel\troute\tknob\tknob_value\tpipeline"
    lines = [header]
    for level, frac, how in (("half", 0.5, "thin"), ("quarter", 0.25, "thin"), ("prefix", 0.5, "first")):
        keep = int(round(frac * n))
        mask = (thin_reads(genome["rng"], genome["reads"], keep) if how == "thin"
                else first_reads(genome["reads"], keep))
        pid = f"{track}__depth__{level}"
        _write_npz(cf / "products" / pid / "counts25.npz",
                   counts_of(genome["reads"], mask, genome["length"]))
        lines.append(f"{pid}\tCF_C19__depth__{level}\t{track}\tC19\tH3K27ac\tdepth\t{level}\tbam\t"
                     f"treatment_reads\t{keep}\tchip")
    rows = tmp_path / "rows.tsv"
    rows.write_text("\n".join(lines) + "\n")

    recs = checks.run(cf, rows, only="depth_law")
    assert {r["pid"]: r["pass"] for r in recs} == {
        f"{track}__depth__half": True, f"{track}__depth__quarter": True,
        f"{track}__depth__prefix": False}
    rec = json.loads((cf / "checks" / "depth_law" / f"{track}__depth__quarter.json").read_text())
    assert set(rec) == {"name", "pid", "pass", "detail"}
    assert rec["detail"]["base_depth"] == n and abs(rec["detail"]["p"] - 0.25) < 1e-6
    out = checks.summary(cf)
    assert out["all_pass"] is False and out["n_checks"] == 3
    assert [(f["name"], f["pid"]) for f in out["failures"]] == [("depth_law", f"{track}__depth__prefix")]


# --- the full tree: every other check on real C1 rows ---------------------------------------------

STUB_RECORDS = '''
import argparse, csv, pathlib, sys
ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd")
v = sub.add_parser("validate")
v.add_argument("--products"); v.add_argument("--rows"); v.add_argument("--expect", type=int)
a = ap.parse_args()
pids = [r["pid"] for r in csv.DictReader(open(a.rows), delimiter="\\t")]
bad = [p for p in pids if not (pathlib.Path(a.products) / p / "covariates.json").exists()]
for p in bad:
    print(f"{p}: covariates.json missing")
if bad or (a.expect is not None and len(pids) != a.expect):
    sys.exit(1)
print(f"OK {len(pids)}")
'''

CHROM_LEN = {c: 1000 for c in [f"chr{i}" for i in range(1, 23)] + ["chrX"]}
CHROM_LEN["chr21"] = 1013   # a remainder, so n_bins == len // 25 is actually exercised


@pytest.fixture()
def tree(checks, tmp_path, monkeypatch):
    """A clean product tree for the two C19 tracks (real `arms.py rows`), depth rows left out."""
    stub = tmp_path / "records_stub.py"
    stub.write_text(STUB_RECORDS)
    monkeypatch.setattr(checks, "RECORDS_PY", stub)
    chrsz = tmp_path / "chrom.sizes.tsv"
    chrsz.write_text("".join(f"{c}\t{n}\n" for c, n in CHROM_LEN.items()) + "chrEBV\t171823\n")

    out = subprocess.run([sys.executable, str(ARMS), "rows", "--route", "all", "--dnase", "atac",
                          "--ratio", "yes", "--tracks", "C19M16,C19M22,C12M02"],
                         capture_output=True, text=True, check=True).stdout
    header, *lines = out.rstrip("\n").split("\n")
    lines = [ln for ln in lines if ln.split("\t")[5] != "depth"]
    rows = tmp_path / "rows_all.tsv"
    rows.write_text("\n".join([header, *lines]) + "\n")
    parsed = [dict(zip(header.split("\t"), ln.split("\t"))) for ln in lines]

    rng = np.random.default_rng(7)
    cf = tmp_path / "cf"
    counts = lambda: {c: rng.integers(0, 50, n // RES).astype(np.uint32) for c, n in CHROM_LEN.items()}
    pval = lambda: {c: rng.random(n // RES).astype(np.float32) * 5 for c, n in CHROM_LEN.items()}
    base_counts, fastq_ctl = {}, {}
    for r in parsed:            # base rows come first per track in C1 order
        pdir = cf / "products" / r["pid"]
        if r["arm"] == "base":
            base_counts[r["track"]] = counts()
        c = base_counts[r["track"]] if r["arm"] in ("base", *checks.P_ONLY_ARMS) else counts()
        _write_npz(pdir / "counts25.npz", c)
        p = pval()
        _write_npz(pdir / "pval25.npz", p)
        if r["arm"] == "base":
            _write_npz(cf / "bamarms" / r["pid"] / "rebuild_pval25.npz", p)
        if r["pipeline"] == "chip":
            if r["route"] == "fastq":
                ctl = fastq_ctl.setdefault(r["biosample"], counts())
            else:
                ctl = counts()
            _write_npz(pdir / "control_counts25.npz", ctl)
        (pdir / "covariates.json").write_text(json.dumps({"pid": r["pid"], "depth": 30000000}))
    (cf / "smoke" / "C8").mkdir(parents=True)
    (cf / "smoke" / "C8" / "ratio_k1.json").write_text(json.dumps({"bit_identical": True}))
    return {"cf": cf, "rows": rows, "chrsz": chrsz, "parsed": parsed}


def _counts_by_name(recs):
    out = {}
    for r in recs:
        out.setdefault(r["name"], []).append(r["pass"])
    return {k: (sum(v), len(v)) for k, v in out.items()}


def test_read_rows_keeps_json_string_knob_values(checks, tree):
    kv = {r["pid"]: r["knob_value"] for r in checks.read_rows(tree["rows"])}
    assert kv["C19M16__ctlid__other"] == "ENCFF337JNL" and kv["C19M16__ctlid__none"] == "none"
    assert kv["C19M16__base__base"] is None and kv["C19M16__pe__pe"] is True
    assert kv["C19M16__crop__36"] == 36


def test_clean_tree_all_pass(checks, tree):
    recs = checks.run(tree["cf"], tree["rows"], chrsz=tree["chrsz"])
    n_rows = len(tree["parsed"])
    # C19M16 + C19M22: 1 base each (+ C12M02 base); P-only arms: (ratio 2 + ctlid 2 + ctldepth 2 +
    # extsize 2) per ChIP track, extsize 2 for DNase; ChIP fastq biosamples (pe, dedup, crop×2,
    # mapq×2) = 6 shared by the two C19 tracks; one ratio k=1 smoke record.
    assert _counts_by_name(recs) == {
        "base_rebuild": (3, 3), "counts_identity": (18, 18), "ratio_k1_identity": (1, 1),
        "fastq_control_identity": (6, 6), "structure": (n_rows, n_rows)}
    out = checks.summary(tree["cf"])
    assert out["all_pass"] is True and out["failures"] == [] and out["n_checks"] == 28 + n_rows
    on_disk = json.loads((tree["cf"] / "checks" / "summary.json").read_text())
    assert on_disk["all_pass"] is True and on_disk["failures"] == []
    md = (tree["cf"] / "checks" / "CHECKS.md").read_text()
    assert "| counts_identity | 18 | 18 |" in md and "| depth_law | 0 | 0 |" in md


def _npz_edit(path, fn):
    arrs = dict(np.load(path))
    fn(arrs)
    np.savez_compressed(path, **arrs)


def _bump(chrom, delta, i=3):
    def f(a):
        a[chrom] = a[chrom].copy()
        a[chrom][i] += delta
    return f


DEFECTS = {
    "rebuild_off_by_1e-3": (
        lambda cf: _npz_edit(cf / "bamarms/C19M16__base__base/rebuild_pval25.npz",
                             _bump("chr21", np.float32(1e-3))),
        [("base_rebuild", "C19M16__base__base")]),
    "rebuild_short_grid": (
        lambda cf: _npz_edit(cf / "bamarms/C12M02__base__base/rebuild_pval25.npz",
                             lambda a: a.__setitem__("chr21", a["chr21"][:-1])),
        [("base_rebuild", "C12M02__base__base")]),
    "rebuild_missing": (
        lambda cf: (cf / "bamarms/C19M22__base__base/rebuild_pval25.npz").unlink(),
        [("base_rebuild", "C19M22__base__base")]),
    "extsize_counts_moved": (
        lambda cf: _npz_edit(cf / "products/C19M22__extsize__k2/counts25.npz", _bump("chrX", 1)),
        [("counts_identity", "C19M22__extsize__k2")]),
    "ratio_k1_not_identical": (
        lambda cf: (cf / "smoke/C8/ratio_k1.json").write_text(json.dumps({"bit_identical": False})),
        [("ratio_k1_identity", "C19M16__ratio__k1")]),
    "fastq_control_differs": (
        lambda cf: _npz_edit(cf / "products/C19M22__mapq__10/control_counts25.npz", _bump("chr1", 1)),
        [("fastq_control_identity", "CF_C19__mapq__10")]),
    "pval_negative": (
        lambda cf: _npz_edit(cf / "products/C19M16__ctlid__none/pval25.npz", _bump("chr2", -100)),
        [("structure", "C19M16__ctlid__none")]),
    "pval_nan": (
        lambda cf: _npz_edit(cf / "products/C12M02__pe__pe/pval25.npz", _bump("chr5", np.nan)),
        [("structure", "C12M02__pe__pe")]),
    "counts_wrong_dtype": (
        lambda cf: _npz_edit(cf / "products/C19M16__crop__36/counts25.npz",
                             lambda a: a.__setitem__("chr3", a["chr3"].astype(np.int64))),
        [("structure", "C19M16__crop__36")]),
    "chrom_dropped": (
        lambda cf: _npz_edit(cf / "products/C19M16__abproxy__f0.5/counts25.npz",
                             lambda a: a.pop("chr7")),
        [("structure", "C19M16__abproxy__f0.5")]),
    "records_validate_fails": (
        lambda cf: (cf / "products/C19M22__dedup__off/covariates.json").unlink(),
        [("structure", "C19M22__dedup__off")]),
}


@pytest.mark.parametrize("defect", list(DEFECTS))
def test_each_defect_fails_exactly_its_check(checks, tree, defect):
    apply, expected = DEFECTS[defect]
    apply(tree["cf"])
    checks.run(tree["cf"], tree["rows"], chrsz=tree["chrsz"])
    out = checks.summary(tree["cf"])
    assert out["all_pass"] is False
    assert sorted((f["name"], f["pid"]) for f in out["failures"]) == sorted(expected)
    rec = json.loads((tree["cf"] / "checks" / expected[0][0] / f"{expected[0][1]}.json").read_text())
    assert rec["pass"] is False


def test_rebuild_within_tolerance_passes(checks, tree):
    _npz_edit(tree["cf"] / "bamarms/C19M16__base__base/rebuild_pval25.npz",
              lambda a: a.__setitem__("chr21", a["chr21"] + np.float32(5e-7)))
    ok, detail = checks.check_base_rebuild(tree["cf"], "C19M16__base__base")
    assert ok, detail


def test_killed_run_leaves_missing_records_as_failures(checks, tree):
    checks.run(tree["cf"], tree["rows"], chrsz=tree["chrsz"])
    (tree["cf"] / "checks" / "structure" / "C19M16__pe__pe.json").unlink()
    out = checks.summary(tree["cf"])
    assert out["all_pass"] is False
    assert [(f["name"], f["pid"]) for f in out["failures"]] == [("structure", "C19M16__pe__pe")]


def test_rerun_replaces_a_stale_record(checks, tree):
    stale = tree["cf"] / "checks" / "counts_identity" / "GONE__ratio__k2.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"name": "counts_identity", "pid": "GONE__ratio__k2",
                                 "pass": False, "detail": {}}))
    checks.run(tree["cf"], tree["rows"], only="counts_identity")
    assert not stale.exists()


def test_cli_run_then_summary(tree):
    cf, rows = str(tree["cf"]), str(tree["rows"])
    r = subprocess.run([sys.executable, str(TOOL), "run", "--cf", cf, "--rows", rows,
                        "--only", "counts_identity"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "18 checks, 0 failed"
    s = subprocess.run([sys.executable, str(TOOL), "summary", "--cf", cf],
                       capture_output=True, text=True)
    assert s.returncode == 0, s.stderr
    d = json.load(open(Path(cf) / "checks" / "summary.json"))
    assert d["all_pass"] is True and len(d["failures"]) == 0
    assert set(d) == {"all_pass", "n_checks", "failures", "created_utc"}
