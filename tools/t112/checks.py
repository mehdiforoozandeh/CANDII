"""t112 step 5 — the checks every counterfactual product must pass before anything reads it.

The task body (`cruxvault/tasks/t112_build_the_counterfactual_arms_for_t.md`, Plan step 5) asks for
three things; this module adds three structural ones the plan pins beside them:

    base_rebuild            base pval25 rebuilt from the kept tagAligns == the pipeline's own
                            bigwig binned the same way, chr21, same n_bins, max |diff| <= 1e-6
    counts_identity         arms that only move p (ratio, ctlid, ctldepth, extsize) carry counts
                            bit-identical to their base, all 23 chromosomes
    depth_law               a depth arm reproduces the store's `thin_counts` law in expectation:
                            conditional on base == k, arm ~ Binomial(k, p), p = L / base depth
    ratio_k1_identity       the one-token `--ratio` patch at k = 1 is bit-identical to the
                            unpatched script (C8 smoke record)
    fastq_control_identity  a FASTQ re-run re-aligns the control; the two tracks of one cell must
                            still land on the same control counts
    structure               records.py validate, 23 chromosomes, n_bins == len // 25, counts
                            uint32, pval float32, finite, >= 0

`src/candi/store/dataset.py::thin_counts` is `rng.binomial(counts, 1/d)` per bin. The depth arms
are made by thinning *reads* with the pipeline's own subsampler, not bins, so the law only holds
to first order — hence tolerances, not equality. The thresholds (0.005 on the total fraction, 2 %
on the conditional mean, 0.85–1.15 on the variance ratio, k <= 10, >= 10000 base bins) are
planner-set data checks. They are not an experiment gate, and they are not to be tuned to make a
product pass.

Every check writes `<cf>/checks/<name>/<pid>.json` = `{name, pid, pass, detail}`; `summary` rolls
them into `<cf>/checks/summary.json` and `<cf>/checks/CHECKS.md`. `run` first writes the (check, pid)
list it is about to do to `<cf>/checks/expected/<name>.json`, so a job killed half way leaves
records missing, and `summary` counts each missing record as a failure rather than a smaller total.

numpy + pandas (pandas only because C2's `bin25.py` imports it) and no `candi` import: it runs on
Nibi against a code snapshot. The npz reader, chrom-sizes parser and chromosome set are C2's
(`bin25.py`), and product validation is C3's (`records.py validate`), both siblings here.

    python checks.py run --cf /scratch/mforooz/t112_cf --rows checks/rows_all.tsv [--only NAME]
    python checks.py summary --cf /scratch/mforooz/t112_cf
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bin25  # noqa: E402  (C2, sibling)

#: C3's validator, run as its pinned CLI. A module attribute so tests can point it at a stub.
RECORDS_PY = HERE / "records.py"

#: the pipeline's chrom sizes on Nibi (plan shorthand CHRSZ).
CHRSZ = "/scratch/mforooz/EIC_REPRO/003/refcache/c52f52c7bfa357f55a39b1de7e4d0b0c/GRCh38_EBV.chrom.sizes.tsv"

MAIN_CHROMS = bin25.MAIN_CHROMS
RES = 25

CHECK_NAMES = ("base_rebuild", "counts_identity", "depth_law", "ratio_k1_identity",
               "fastq_control_identity", "structure")

#: arms whose knob sits after counting: counts must equal base bit for bit.
P_ONLY_ARMS = ("ratio", "ctlid", "ctldepth", "extsize")

# planner-set thresholds (plan Log, 2026-09-17). Do not tune.
BASE_REBUILD_CHROM = "chr21"
BASE_REBUILD_TOL = 1e-6
DEPTH_TOTAL_TOL = 0.005
DEPTH_MEAN_TOL = 0.02
DEPTH_VAR_LO, DEPTH_VAR_HI = 0.85, 1.15
DEPTH_K_MAX = 10
DEPTH_MIN_BINS = 10000


def read_npz(path: Path) -> dict:
    return bin25.read_npz(path)


def load_chrsz(path) -> dict:
    return bin25.load_chrsz(path)


def read_rows(path: Path) -> list:
    """A C1 `arms.py rows` TSV; `knob_value` is JSON text (QUOTE_NONE, or csv eats its quotes)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    for r in rows:
        r["knob_value"] = json.loads(r["knob_value"]) if r["knob_value"] != "" else None
    return rows


def base_pid(row: dict) -> str:
    return f"{row['track']}__base__base"


def _num(x):
    """JSON-safe float: NaN / inf become strings rather than invalid JSON."""
    x = float(x)
    return x if math.isfinite(x) else str(x)


def write_record(cf: Path, name: str, pid: str, ok: bool, detail: dict) -> dict:
    rec = {"name": name, "pid": pid, "pass": bool(ok), "detail": detail}
    out = Path(cf) / "checks" / name / f"{pid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


# ---------------------------------------------------------------------------------------------
# the checks: each returns (pass, detail) for one pid, and raises if an input is missing
# ---------------------------------------------------------------------------------------------


def check_base_rebuild(cf: Path, pid: str):
    pipe = read_npz(cf / "products" / pid / "pval25.npz")[BASE_REBUILD_CHROM]
    rebuilt = read_npz(cf / "bamarms" / pid / "rebuild_pval25.npz")[BASE_REBUILD_CHROM]
    detail = {"chrom": BASE_REBUILD_CHROM, "n_bins_pipeline": int(pipe.size),
              "n_bins_rebuild": int(rebuilt.size), "tol": BASE_REBUILD_TOL}
    if pipe.size != rebuilt.size:
        return False, detail
    a, b = pipe.astype(np.float64), rebuilt.astype(np.float64)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    detail["n_nan_pipeline"], detail["n_nan_rebuild"] = int(nan_a.sum()), int(nan_b.sum())
    same_nan = bool(np.array_equal(nan_a, nan_b))
    ok = ~nan_a & ~nan_b
    mad = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0
    detail["nan_mask_equal"] = same_nan
    detail["max_abs_diff"] = _num(mad)
    return same_nan and mad <= BASE_REBUILD_TOL, detail


def check_counts_identity(cf: Path, pid: str, base: str):
    arm = read_npz(cf / "products" / pid / "counts25.npz")
    ref = read_npz(cf / "products" / base / "counts25.npz")
    differ = [c for c in MAIN_CHROMS
              if c not in arm or c not in ref or not np.array_equal(arm[c], ref[c])]
    return not differ, {"base": base, "n_chroms": len(MAIN_CHROMS), "differ": differ}


def depth_law(base_counts: dict, arm_counts: dict, p: float):
    """The `thin_counts` law in expectation, over all main chromosomes. Pure; tested directly."""
    sum_b = sum_a = 0
    n = np.zeros(DEPTH_K_MAX + 1, dtype=np.int64)
    s = np.zeros(DEPTH_K_MAX + 1, dtype=np.float64)
    ss = np.zeros(DEPTH_K_MAX + 1, dtype=np.float64)
    missing = [c for c in MAIN_CHROMS if c not in base_counts or c not in arm_counts]
    for c in MAIN_CHROMS:
        if c in missing:
            continue
        b, a = base_counts[c], arm_counts[c]
        if b.shape != a.shape:
            missing.append(c)
            continue
        sum_b += int(b.sum(dtype=np.int64))
        sum_a += int(a.sum(dtype=np.int64))
        sel = (b >= 1) & (b <= DEPTH_K_MAX)
        bk = b[sel].astype(np.int64)
        ak = a[sel].astype(np.float64)
        n += np.bincount(bk, minlength=DEPTH_K_MAX + 1)
        s += np.bincount(bk, weights=ak, minlength=DEPTH_K_MAX + 1)
        ss += np.bincount(bk, weights=ak * ak, minlength=DEPTH_K_MAX + 1)
    frac = sum_a / sum_b if sum_b else float("nan")
    total_ok = sum_b > 0 and abs(frac - p) <= DEPTH_TOTAL_TOL
    per_k, all_k_ok = {}, True
    for k in range(1, DEPTH_K_MAX + 1):
        if n[k] < DEPTH_MIN_BINS:
            continue
        mean = s[k] / n[k]
        var = ss[k] / n[k] - mean * mean
        mean_ratio = mean / (k * p)
        var_ratio = var / (k * p * (1 - p)) if p < 1 else float("nan")
        ok = (abs(mean_ratio - 1) <= DEPTH_MEAN_TOL
              and DEPTH_VAR_LO <= var_ratio <= DEPTH_VAR_HI)
        all_k_ok &= bool(ok)
        per_k[str(k)] = {"n_bins": int(n[k]), "mean_ratio": _num(mean_ratio),
                         "var_ratio": _num(var_ratio), "pass": bool(ok)}
    detail = {"p": _num(p), "sum_base": sum_b, "sum_arm": sum_a, "frac": _num(frac),
              "total_pass": bool(total_ok), "per_k": per_k, "n_k_tested": len(per_k),
              "missing_or_misshapen": missing,
              "thresholds": {"total": DEPTH_TOTAL_TOL, "mean": DEPTH_MEAN_TOL,
                             "var": [DEPTH_VAR_LO, DEPTH_VAR_HI], "k_max": DEPTH_K_MAX,
                             "min_bins": DEPTH_MIN_BINS}}
    ok = total_ok and all_k_ok and not missing and len(per_k) > 0
    return ok, detail


def check_depth_law(cf: Path, pid: str, base: str, reads_kept: int):
    base_cov = json.loads((cf / "products" / base / "covariates.json").read_text())
    p = int(reads_kept) / int(base_cov["depth"])
    ok, detail = depth_law(read_npz(cf / "products" / base / "counts25.npz"),
                           read_npz(cf / "products" / pid / "counts25.npz"), p)
    detail.update(base=base, reads_kept=int(reads_kept), base_depth=int(base_cov["depth"]))
    return ok, detail


def check_ratio_k1(cf: Path):
    path = cf / "smoke" / "C8" / "ratio_k1.json"
    rec = json.loads(path.read_text())
    return rec.get("bit_identical") is True, {"source": str(path), "record": rec}


def check_fastq_control_identity(cf: Path, pids: list):
    arrays = [read_npz(cf / "products" / p / "control_counts25.npz") for p in pids]
    differ = [c for c in MAIN_CHROMS
              if any(c not in a for a in arrays)
              or not all(np.array_equal(arrays[0][c], a[c]) for a in arrays[1:])]
    return not differ, {"pids": pids, "differ": differ}


def check_structure(cf: Path, pid: str, sizes: dict, header: str, line: str):
    problems = []
    with tempfile.TemporaryDirectory() as td:
        # C3's validator on a one-row copy of the rows TSV: its problem lines are then this pid's.
        rows_tsv = Path(td) / "rows.tsv"
        rows_tsv.write_text(header + "\n" + line + "\n")
        res = subprocess.run([sys.executable, str(RECORDS_PY), "validate",
                              "--products", str(cf / "products"), "--rows", str(rows_tsv),
                              "--expect", "1"], capture_output=True, text=True)
    lines = res.stdout.strip().splitlines()
    validate_ok = res.returncode == 0 and bool(lines) and lines[-1].strip() == "OK 1"
    if not validate_ok:
        problems.append({"records_validate": lines[-20:], "stderr": res.stderr.strip()[-2000:],
                         "returncode": res.returncode})
    pdir = cf / "products" / pid
    files = [("counts25.npz", np.uint32), ("pval25.npz", np.float32)]
    if (pdir / "control_counts25.npz").exists():
        files.append(("control_counts25.npz", np.uint32))
    for fname, dtype in files:
        path = pdir / fname
        if not path.exists():
            problems.append(f"{fname}: missing")
            continue
        arrs = read_npz(path)
        if sorted(arrs) != sorted(MAIN_CHROMS):
            problems.append(f"{fname}: chroms {sorted(arrs)} != the 23 main chroms")
        for c in MAIN_CHROMS:
            if c not in arrs:
                continue
            a = arrs[c]
            if a.ndim != 1 or a.size != sizes[c] // RES:
                problems.append(f"{fname}:{c}: shape {a.shape} != ({sizes[c] // RES},)")
            if a.dtype != dtype:
                problems.append(f"{fname}:{c}: dtype {a.dtype} != {np.dtype(dtype)}")
            if a.dtype.kind == "f":
                n_bad = int((~np.isfinite(a)).sum())
                n_neg = int((a < 0).sum())
                if n_bad:
                    problems.append(f"{fname}:{c}: {n_bad} non-finite")
                if n_neg:
                    problems.append(f"{fname}:{c}: {n_neg} negative")
    return not problems, {"records_validate_ok": validate_ok,
                          "files": [f for f, _ in files], "problems": problems}


# ---------------------------------------------------------------------------------------------
# run / summary
# ---------------------------------------------------------------------------------------------


def _guarded(cf, name, pid, fn, *args):
    """A missing or unreadable input is a failed check, not a crashed run."""
    try:
        ok, detail = fn(*args)
    except Exception as e:  # noqa: BLE001 — recorded, not swallowed
        ok, detail = False, {"error": f"{type(e).__name__}: {e}"}
    rec = write_record(cf, name, pid, ok, detail)
    print(f"{'PASS' if ok else 'FAIL'} {name} {pid}", flush=True)
    return rec


def plan(cf, rows_tsv, names, chrsz=CHRSZ) -> list:
    """Every (name, pid, fn, args) the rows call for, for the checks in `names`."""
    cf = Path(cf)
    rows = read_rows(Path(rows_tsv))
    jobs = []
    if "base_rebuild" in names:
        jobs += [("base_rebuild", r["pid"], check_base_rebuild, (cf, r["pid"]))
                 for r in rows if r["arm"] == "base"]
    if "counts_identity" in names:
        jobs += [("counts_identity", r["pid"], check_counts_identity, (cf, r["pid"], base_pid(r)))
                 for r in rows if r["arm"] in P_ONLY_ARMS]
    if "depth_law" in names:
        jobs += [("depth_law", r["pid"], check_depth_law, (cf, r["pid"], base_pid(r), r["knob_value"]))
                 for r in rows if r["arm"] == "depth"]
    if "ratio_k1_identity" in names and any(r["arm"] == "ratio" for r in rows):
        track = next(r["track"] for r in rows if r["arm"] == "ratio")
        jobs.append(("ratio_k1_identity", f"{track}__ratio__k1", check_ratio_k1, (cf,)))
    if "fastq_control_identity" in names:
        groups = {}
        for r in rows:  # the biosample is exactly (cell, arm, level)
            if r["route"] == "fastq" and r["pipeline"] != "atac":
                groups.setdefault(r["biosample"], []).append(r["pid"])
        jobs += [("fastq_control_identity", bios, check_fastq_control_identity, (cf, pids))
                 for bios, pids in groups.items() if len(pids) >= 2]
    if "structure" in names:
        sizes = load_chrsz(chrsz)
        header, *lines = Path(rows_tsv).read_text().splitlines()
        raw = {ln.split("\t", 1)[0]: ln for ln in lines if ln.strip()}
        jobs += [("structure", r["pid"], check_structure, (cf, r["pid"], sizes, header, raw[r["pid"]]))
                 for r in rows]
    return jobs


def run(cf, rows_tsv, only=None, chrsz=CHRSZ) -> list:
    cf = Path(cf)
    names = [only] if only else list(CHECK_NAMES)
    jobs = plan(cf, rows_tsv, names, chrsz)
    for name in names:  # a rerun replaces its records; a stale pass must not survive
        for old in (cf / "checks" / name).glob("*.json"):
            old.unlink()
        exp = cf / "checks" / "expected" / f"{name}.json"
        exp.parent.mkdir(parents=True, exist_ok=True)
        exp.write_text(json.dumps([pid for n, pid, _, _ in jobs if n == name], indent=1) + "\n")
    return [_guarded(cf, name, pid, fn, *args) for name, pid, fn, args in jobs]


def summary(cf) -> dict:
    cf = Path(cf)
    recs = []
    for name in CHECK_NAMES:
        for path in sorted((cf / "checks" / name).glob("*.json")):
            recs.append(json.loads(path.read_text()))
    failures = [{"name": r["name"], "pid": r["pid"], "detail": r["detail"]}
                for r in recs if not r["pass"]]
    expected = {}
    for name in CHECK_NAMES:
        exp = cf / "checks" / "expected" / f"{name}.json"
        expected[name] = json.loads(exp.read_text()) if exp.exists() else []
        have = {r["pid"] for r in recs if r["name"] == name}
        failures += [{"name": name, "pid": pid, "detail": {"error": "no record: the run did not reach it"}}
                     for pid in expected[name] if pid not in have]
    n_checks = sum(len({r["pid"] for r in recs if r["name"] == name} | set(expected[name]))
                   for name in CHECK_NAMES)
    out = {"all_pass": bool(recs) and not failures, "n_checks": n_checks,
           "failures": failures,
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (cf / "checks").mkdir(parents=True, exist_ok=True)
    (cf / "checks" / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")

    md = ["# t112 pre-use checks", "",
          f"All pass: **{out['all_pass']}** ({out['n_checks']} checks, {len(failures)} failures). "
          f"Written {out['created_utc']}.", "",
          "| check | pass | total |", "|---|---|---|"]
    for name in CHECK_NAMES:
        mine = [r for r in recs if r["name"] == name]
        total = len({r["pid"] for r in mine} | set(expected[name]))
        md.append(f"| {name} | {sum(r['pass'] for r in mine)} | {total} |")
    if failures:
        md += ["", "## Failures", ""]
        md += [f"- `{f['name']}` `{f['pid']}`" for f in failures]
    (cf / "checks" / "CHECKS.md").write_text("\n".join(md) + "\n")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="run the checks; one JSON record per (check, pid)")
    pr.add_argument("--cf", required=True, help="the t112 root, e.g. /scratch/mforooz/t112_cf")
    pr.add_argument("--rows", required=True, help="`arms.py rows --route all` TSV")
    pr.add_argument("--only", choices=CHECK_NAMES, default=None)
    pr.add_argument("--chrsz", default=CHRSZ, help="chrom sizes for `structure` (default: CHRSZ)")
    ps = sub.add_parser("summary", help="write checks/summary.json and checks/CHECKS.md")
    ps.add_argument("--cf", required=True)
    args = p.parse_args(argv)
    if args.cmd == "run":
        recs = run(args.cf, args.rows, args.only, args.chrsz)
        n_fail = sum(not r["pass"] for r in recs)
        print(f"{len(recs)} checks, {n_fail} failed")
        return 0
    out = summary(args.cf)
    print(f"all_pass={out['all_pass']} n_checks={out['n_checks']} failures={len(out['failures'])}")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
