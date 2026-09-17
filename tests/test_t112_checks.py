"""t112 C10 — `tools/t112/checks.py`: the step-5 pre-use checks and the structural ones beside them.

The depth-law tests build reads, thin them the way the pipeline's subsampler does (a draw of L reads
without replacement), and bin both sides with the store overlap rule. An honest thinning at
p = 0.125 must pass; a scaled copy rint(base·p), a thinning from a superset of base's reads, and a
thinning from an independent read set must each fail (plan C10 Test line). The thresholds are the
plan's and are checked against its numbers once, never tuned here.

The product tree carries covariates/provenance records valid under C3's pinned schema. `structure`
runs C3's real `tools/t112/records.py` when it is on disk; a branch cut before C3 merged falls back
to a stub honouring C3's pinned CLI (`validate --products D --rows R --expect N` → `OK <n>`).
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "t112" / "checks.py"
ARMS = REPO / "tools" / "t112" / "arms.py"
RECORDS = REPO / "tools" / "t112" / "records.py"
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


@pytest.fixture(scope="module")
def genome(checks):
    rng = np.random.default_rng(112)
    reads, length = make_reads(rng, checks.MAIN_CHROMS, bins_per_chrom=100_000)
    n_reads = sum(s.size for s, _ in reads.values())
    base = counts_of(reads, _all(reads), length)
    return {"reads": reads, "length": length, "n_reads": n_reads, "base": base, "rng": rng}


# --- constants --------------------------------------------------------------------------------------

def test_thresholds_are_the_plan_values(checks):
    assert checks.DEPTH_TOTAL_TOL == 0.005
    assert (checks.DEPTH_D_LO, checks.DEPTH_D_HI) == (0.90, 1.10)
    assert checks.DEPTH_K_MAX == 10
    assert checks.DEPTH_MIN_BINS == 100000
    assert checks.BASE_REBUILD_TOL == 1e-6 and checks.BASE_REBUILD_CHROM == "chr21"
    assert len(checks.MAIN_CHROMS) == 23 and checks.MAIN_CHROMS[-1] == "chrX"
    assert checks.P_ONLY_ARMS == ("ratio", "ctlid", "ctldepth", "extsize")
    assert checks.RATIO_K1_RECORD.count("__") != 2   # cannot parse as track__arm__level


# --- depth_law ----------------------------------------------------------------------------------------

def _all(reads):
    return {c: np.ones(s.size, bool) for c, (s, _) in reads.items()}


@pytest.mark.parametrize("p", [0.125, 0.5])
def test_depth_law_passes_honest_per_read_thinning(checks, genome, p):
    n_keep = int(round(p * genome["n_reads"]))
    arm = counts_of(genome["reads"], thin_reads(genome["rng"], genome["reads"], n_keep),
                    genome["length"])
    ok, detail = checks.depth_law(genome["base"], arm, n_keep / genome["n_reads"])
    assert ok, detail
    assert detail["nested"] and detail["n_bins_violating"] == 0
    assert detail["n_bins"] >= 100000 and 0.9 <= detail["D"] <= 1.1
    assert abs(detail["frac"] - p) <= 0.005


def test_depth_law_fails_scaled_copy(checks, genome):
    """rint(base·p): nested, but no Binomial spread (D = 0) and the wrong total at p = 0.125."""
    p = 0.125
    arm = {c: np.rint(b * p).astype(np.uint32) for c, b in genome["base"].items()}
    ok, detail = checks.depth_law(genome["base"], arm, p)
    assert not ok
    assert detail["nested"] and not detail["D_pass"]


def test_depth_law_fails_thinning_from_a_superset(checks, genome):
    """Base is itself a 1/1.5 draw of the reads; the arm is drawn from all of them, not from base."""
    p, rng, reads = 0.125, genome["rng"], genome["reads"]
    n_base = int(round(genome["n_reads"] / 1.5))
    base = counts_of(reads, thin_reads(rng, reads, n_base), genome["length"])
    n_arm = int(round(p * n_base))
    arm = counts_of(reads, thin_reads(rng, reads, n_arm), genome["length"])
    ok, detail = checks.depth_law(base, arm, n_arm / n_base)
    assert not ok
    assert not detail["nested"] and detail["n_bins_violating"] > 0


def test_depth_law_fails_thinning_from_an_independent_read_set(checks, genome):
    p = 0.125
    rng = np.random.default_rng(2112)
    other, length = make_reads(rng, checks.MAIN_CHROMS, bins_per_chrom=100_000)
    n_other = sum(s.size for s, _ in other.values())
    arm = counts_of(other, thin_reads(rng, other, int(round(p * n_other))), length)
    ok, detail = checks.depth_law(genome["base"], arm, p)
    assert not ok
    assert not detail["nested"] and detail["n_bins_violating"] > 0


def test_depth_law_fails_on_wrong_p_and_on_missing_chrom(checks, genome):
    n_keep = int(round(0.125 * genome["n_reads"]))
    arm = counts_of(genome["reads"], thin_reads(genome["rng"], genome["reads"], n_keep),
                    genome["length"])
    ok, detail = checks.depth_law(genome["base"], arm, 0.25)   # claimed level != drawn level
    assert not ok and not detail["total_pass"]
    short = dict(arm)
    del short["chr21"]
    ok, detail = checks.depth_law(genome["base"], short, n_keep / genome["n_reads"])
    assert not ok and not detail["nested"] and detail["missing_or_misshapen"] == ["chr21"]


def test_depth_law_fails_with_too_few_bins(checks):
    """An honest thinning, but only 23 × 4000 bins with base in 1..10 (< 100000)."""
    rng = np.random.default_rng(5)
    reads, length = make_reads(rng, checks.MAIN_CHROMS, bins_per_chrom=4000, cover=6.0)
    n = sum(s.size for s, _ in reads.values())
    base = counts_of(reads, _all(reads), length)
    n_keep = int(round(0.5 * n))
    ok, detail = checks.depth_law(base, counts_of(reads, thin_reads(rng, reads, n_keep), length),
                                  n_keep / n)
    assert not ok and detail["n_bins"] < 100000 and not detail["n_bins_pass"]
    assert detail["nested"]


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
    for level, frac, how in (("half", 0.5, "thin"), ("eighth", 0.125, "thin"),
                             ("superset", 0.125, "superset")):
        keep = int(round(frac * n))
        if how == "thin":
            mask = thin_reads(genome["rng"], genome["reads"], keep)
        else:  # drawn from a larger read set than base holds: base plus an extra 10 %
            extra, _ = make_reads(np.random.default_rng(9), checks.MAIN_CHROMS, 100_000, cover=0.2)
            both = {c: (np.concatenate([genome["reads"][c][0], extra[c][0]]),
                        np.concatenate([genome["reads"][c][1], extra[c][1]]))
                    for c in checks.MAIN_CHROMS}
            _write_npz(cf / "products" / f"{track}__depth__{level}" / "counts25.npz",
                       counts_of(both, thin_reads(genome["rng"], both, keep), genome["length"]))
            mask = None
        pid = f"{track}__depth__{level}"
        if mask is not None:
            _write_npz(cf / "products" / pid / "counts25.npz",
                       counts_of(genome["reads"], mask, genome["length"]))
        lines.append(f"{pid}\tCF_C19__depth__{level}\t{track}\tC19\tH3K27ac\tdepth\t{level}\tbam\t"
                     f"treatment_reads\t{keep}\tchip")
    rows = tmp_path / "rows.tsv"
    rows.write_text("\n".join(lines) + "\n")

    recs = checks.run(cf, rows, only="depth_law")
    assert {r["pid"]: r["pass"] for r in recs} == {
        f"{track}__depth__half": True, f"{track}__depth__eighth": True,
        f"{track}__depth__superset": False}
    rec = json.loads((cf / "checks" / "depth_law" / f"{track}__depth__eighth.json").read_text())
    assert set(rec) == {"name", "pid", "pass", "detail"}
    d = rec["detail"]
    assert d["base_depth"] == n and abs(d["p"] - 0.125) < 1e-6
    assert {"nested", "n_bins_violating", "frac", "D", "n_bins"} <= set(d)
    out = checks.summary(cf)
    assert out["all_pass"] is False
    depth_fail = [(f["name"], f["pid"]) for f in out["failures"] if f["name"] == "depth_law"]
    assert depth_fail == [("depth_law", f"{track}__depth__superset")]
    # the other five checks did not run on this tree: each is a failure, not a silent pass
    assert sorted(f["name"] for f in out["failures"] if f["pid"] == "*") == sorted(
        set(checks.CHECK_NAMES) - {"depth_law"})


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

# The pinned C3 schema (plan C3 Interfaces), typed in so the tree is valid for the real validator.
COUNT_RULE = ("store overlap rule: +1 to bins floor(start/25)..floor(end/25) inclusive, "
              "grid floor(len/25)")
WORDING_NOTE = ("t112 says read-start counts; products use the store overlap rule so the loader "
                "reads them unchanged")
PIPE = {"chip": ("ENCODE-DCC/chip-seq-pipeline2", "v2.2.2", "f6e408f7e77bafde4556883f33552191"),
        "atac": ("ENCODE-DCC/atac-seq-pipeline", "v2.2.3", "04d9fa482d67cf633845653750f58cef")}
MATCHED = {"C19": ("ENCFF433TZR", 58073570)}


def write_records(pdir, row, has_control):
    kv = json.loads(row["knob_value"])
    control = None
    if has_control:
        acc, reads = MATCHED[row["cell"]]
        source = "matched"
        if row["arm"] == "ctlid":
            acc, source = kv, "other"
        control = {"accession": acc, "source": source, "reads": reads}
    depth = 50000000 if row["pipeline"] == "atac" else 30000000
    cov = {"schema": 1, "pid": row["pid"], "biosample": row["biosample"], "track": row["track"],
           "cell": row["cell"], "assay": row["assay"], "arm": row["arm"], "level": row["level"],
           "knob": row["knob"], "knob_value": kv, "depth": depth, "log2_depth": math.log2(depth),
           "read_length": 76 if row["pipeline"] == "atac" else 101,
           "run_type": "paired-ended" if row["arm"] == "pe" else "single-ended",
           "fraglen": 150 if row["pipeline"] == "atac" else 180, "control": control,
           "count_rule": COUNT_RULE, "wording_note": WORDING_NOTE}
    repo, release, md5 = PIPE[row["pipeline"]]
    fastq = row["route"] == "fastq"
    prov = {"schema": 1, "pid": row["pid"], "route": row["route"],
            "pipeline": {"repo": repo, "release": release, "sif": "/x/pipeline.sif",
                         "sif_md5": md5, "sif_sha256": None},
            "commands": ["true"], "inputs": [{"path": "/x/in", "md5": "0" * 32}],
            "outputs": [{"path": "/x/out", "md5": "1" * 32}], "subsample_seed": [],
            "patched_script": None,
            "caper": ({"input_json": "/x/i.json", "workflow_id": "w1", "metadata_json": "/x/m.json"}
                      if fastq else None),
            "slurm_job_ids": ["123"], "code": {"snapshot_dir": "/x/code", "git_sha": "abcdef1"},
            "created_utc": "2026-09-17T00:00:00+00:00"}
    (pdir / "covariates.json").write_text(json.dumps(cov))
    (pdir / "provenance.json").write_text(json.dumps(prov))


CHROM_LEN = {c: 1000 for c in [f"chr{i}" for i in range(1, 23)] + ["chrX"]}
CHROM_LEN["chr21"] = 1013   # a remainder, so n_bins == len // 25 is actually exercised


@pytest.fixture()
def tree(checks, tmp_path, monkeypatch):
    """A clean product tree for C19M16, C19M22 and C12M02 (real `arms.py rows`), depth rows left out."""
    if not RECORDS.exists():   # branch cut before C3 merged: the stub honours C3's pinned CLI
        stub = tmp_path / "records_stub.py"
        stub.write_text(STUB_RECORDS)
        monkeypatch.setattr(checks, "RECORDS_PY", stub)
    else:
        assert checks.RECORDS_PY == RECORDS
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
        has_control = r["pipeline"] == "chip" and not (r["arm"] == "ctlid" and r["level"] == "none")
        if has_control:
            if r["route"] == "fastq":
                ctl = fastq_ctl.setdefault(r["biosample"], counts())
            else:
                ctl = counts()
            _write_npz(pdir / "control_counts25.npz", ctl)
        write_records(pdir, r, has_control)
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
    "rebuild_nan": (
        lambda cf: _npz_edit(cf / "bamarms/C19M16__base__base/rebuild_pval25.npz",
                             _bump("chr21", np.float32(np.nan))),
        [("base_rebuild", "C19M16__base__base")]),
    "ratio_k1_not_identical": (
        lambda cf: (cf / "smoke/C8/ratio_k1.json").write_text(json.dumps({"bit_identical": False})),
        [("ratio_k1_identity", "smoke_C8_ratio_k1")]),
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
    "control_npz_without_control": (      # only the real C3 validator can see this one
        lambda cf: (cf / "products/C19M16__ctlid__none/control_counts25.npz").write_bytes(
            (cf / "products/C19M16__base__base/control_counts25.npz").read_bytes()),
        [("structure", "C19M16__ctlid__none")]),
}


@pytest.mark.parametrize("defect", list(DEFECTS))
def test_each_defect_fails_exactly_its_check(checks, tree, defect):
    if defect == "control_npz_without_control" and not RECORDS.exists():
        pytest.skip("needs C3's real records.py (the stub checks only that covariates.json exists)")
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


def test_partial_run_is_not_a_pass(checks, tree):
    """`run --only counts_identity` on a fresh checks/ must not let summary say all_pass."""
    recs = checks.run(tree["cf"], tree["rows"], only="counts_identity")
    assert len(recs) == 18 and all(r["pass"] for r in recs)
    out = checks.summary(tree["cf"])
    assert out["all_pass"] is False
    assert sorted(f["name"] for f in out["failures"]) == sorted(
        set(checks.CHECK_NAMES) - {"counts_identity"})
    assert all(f["pid"] == "*" for f in out["failures"])


def test_run_that_fails_to_plan_leaves_no_stale_pass(checks, tree):
    """A full pass, then a corrupted product and a run that raises while planning (no chrom sizes):
    the old records must be gone, so summary fails."""
    checks.run(tree["cf"], tree["rows"], chrsz=tree["chrsz"])
    assert checks.summary(tree["cf"])["all_pass"] is True
    _npz_edit(tree["cf"] / "products/C19M16__ratio__k2/counts25.npz", _bump("chr4", 1))
    with pytest.raises(FileNotFoundError):
        checks.run(tree["cf"], tree["rows"], chrsz=tree["cf"] / "no_such.chrom.sizes")
    out = checks.summary(tree["cf"])
    assert out["all_pass"] is False
    assert sorted(f["name"] for f in out["failures"]) == sorted(checks.CHECK_NAMES)
    assert not list((tree["cf"] / "checks" / "counts_identity").glob("*.json"))


def test_cli_run_then_summary(tree):
    cf, rows = str(tree["cf"]), str(tree["rows"])
    r = subprocess.run([sys.executable, str(TOOL), "run", "--cf", cf, "--rows", rows,
                        "--chrsz", str(tree["chrsz"])], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    s = subprocess.run([sys.executable, str(TOOL), "summary", "--cf", cf],
                       capture_output=True, text=True)
    d = json.load(open(Path(cf) / "checks" / "summary.json"))
    assert set(d) == {"all_pass", "n_checks", "failures", "created_utc"}
    if RECORDS.exists():   # the CLI runs the real sibling records.py; no stub can be injected
        assert s.returncode == 0, s.stdout + s.stderr
        assert d["all_pass"] is True and len(d["failures"]) == 0
    else:
        assert s.returncode == 1 and d["all_pass"] is False
        assert {f["name"] for f in d["failures"]} == {"structure"}
    r = subprocess.run([sys.executable, str(TOOL), "run", "--cf", cf, "--rows", rows,
                        "--only", "counts_identity"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "18 checks, 0 failed"
