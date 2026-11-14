#!/usr/bin/env python3
"""Build list of the second experiments (t262, chunk OXb of the onion plan, Amendment 4; PI ruling 33).

    other_build_list.py build --selection OXL --catalog DIR --out BX [--batches BATCHES.tsv]
    other_build_list.py check --build-list BX --selection OXL --ledger L --eic-build-list B

`OXL` is the reviewed list of chunk OXs (`ONION/eic/other_experiment_ledger.tsv`): one row per EIC
track with its selected second experiment and a fallback. `DIR` is the D1 ENCODE catalog (its
`files.tsv`). The output has the 15 columns of `build_list.BUILD_LIST_COLUMNS`, so the EIC build
driver (`slurm/onion_data/build_row.sh`) builds it unchanged.

Rows. One row per selected experiment, numbered from 2001 upward, `priority = 5`; one row per
fallback, numbered from 3001 upward, `priority = 6` (never submitted unless the collect chunk asks).
Row numbers are global on the cluster (`build_row.sh` reads the active rows from every onion_*
array task), so they must not meet the EIC build list (rows 1 to 1217). Rows keep the order of
`OXL`. Every row: `role = evalonly`, `eic_track` of the EIC track, `unit = pooled`, `arm = base`,
`control_experiment` empty, `keep_bam = 0`, products `counts+pr1+pr2` of the pooled base view only
(no signal, no control: decision 4; no per-replicate view, no arm).

Files. The FASTQs of a row are `build_list.available_fastqs` of the catalog's FASTQs of the
experiment, with the files that `OXL` lists as served taken as the probe (OXs sent the HEADs). The
result must equal the `OXL` list, else `build` stops. An experiment that `OXL` marks
`source = redacted_bam` (PI ruling 16 extended) is built through the native route as the EIC native
rows are: `pipeline = native-redacted-bam`, `fastq_files` = the BAM accessions of
`build_list.native_bams`, and the measurement is `<experiment>.pooled.native.base`, because
`align_views.config_from_row` requires the layer of every expected measurement to be the row's
layer (`native` for that pipeline). Every other row is `<experiment>.pooled.uniform.base`.

`check` prints five lines, each a count that must be 0, and exits 1 when any is not:
rows outside the reviewed list (experiment, track, priority or row number not as `OXL` and the
numbering rule give them), experiments that are EIC experiments or EIC controls (any ledger column
naming an experiment or a control, and every experiment and control of the EIC build list), row
numbers shared with the EIC build list, rows with a role other than evalonly, and rows that request
signal or name a control.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_data import build_list as B  # noqa: E402
from onion_data import other_experiments as OX  # noqa: E402

ROLE = "evalonly"
FIRST_ROW = {"selected": 2001, "fallback": 3001}
PRIORITY = {"selected": "5", "fallback": "6"}
BATCH_GB = 1800.0
BATCH_COLUMNS = ("batch", "rows", "n_rows", "fastq_gb")


def _sel_files(s: dict, kind: str) -> tuple[str, list[str]]:
    """(source, listed accessions) of the selected or the fallback experiment of one OXL row."""
    pre = "" if kind == "selected" else "fallback_"
    source = s.get(f"{pre}source", "") or "fastq"
    col = f"{pre}alignment_files" if source == "redacted_bam" else f"{pre}fastq_files"
    return source, B.split_acc(s.get(col, ""))


def row_of(s: dict, kind: str, by_exp: dict) -> dict:
    """One build-list row (without its number) for the selected or the fallback experiment."""
    x = s["experiment"] if kind == "selected" else s["fallback"]
    source, listed = _sel_files(s, kind)
    files = by_exp.get(x, [])
    probe = {a: {"http": "200"} for a in listed}
    mixed = []
    if source == "redacted_bam":
        kept = B.native_bams(files, probe)
        layer, pipeline = "native", B.NATIVE_PIPELINE
    else:
        kept, _ = B.available_fastqs(files, probe)
        mixed = B.facts(kept)["mixed_rep"]
        layer, pipeline = "uniform", B.PIPELINE[s["assay_class"]]
    got = [f["file"] for f in kept]
    if sorted(got) != sorted(listed):
        raise SystemExit(f"{s['eic_track']} {kind} {x}: catalog files {got} != the reviewed "
                         f"list's {listed}")
    if not got:
        raise SystemExit(f"{s['eic_track']} {kind} {x}: no file to build from")
    return {"priority": PRIORITY[kind], "experiment": x, "eic_track": s["eic_track"],
            "role": ROLE, "assay": s["assay"], "assay_class": s["assay_class"],
            "unit": "pooled", "arm": "base", "fastq_files": ";".join(got),
            "fastq_gb": f"{B.gb(kept):.3f}", "control_experiment": "", "pipeline": pipeline,
            "keep_bam": "0",
            "expected_measurements": B.token(f"{x}.pooled.{layer}.base", B.BASE_PRODUCTS),
            "_mixed_rep": mixed}


def build(selection: list[dict], catalog_files: list[dict]) -> list[dict]:
    want = {s["experiment"] for s in selection} | {s["fallback"] for s in selection} - {""}
    by_exp = {}
    for f in catalog_files:
        if f["experiment"] in want and (f.get("file_format", "fastq") == "fastq"
                                        or B.is_native_bam(f)):
            by_exp.setdefault(f["experiment"], []).append(f)
    rows = []
    for kind in ("selected", "fallback"):
        n = FIRST_ROW[kind]
        for s in selection:
            if kind == "fallback" and not s.get("fallback"):
                continue
            r = row_of(s, kind, by_exp)
            r["row"] = str(n)
            rows.append(r)
            n += 1
    return rows


def batches(rows: list[dict], cap_gb: float = BATCH_GB) -> list[dict]:
    """The priority-5 rows, in row order, cut into batches of at most `cap_gb` of fastq_gb."""
    out, cur, g = [], [], 0.0
    for r in sorted((r for r in rows if r["priority"] == PRIORITY["selected"]),
                    key=lambda r: int(r["row"])):
        x = float(r["fastq_gb"])
        if cur and g + x > cap_gb:
            out.append((cur, g))
            cur, g = [], 0.0
        cur.append(int(r["row"]))
        g += x
    if cur:
        out.append((cur, g))
    return [{"batch": str(i), "rows": B.spec_of(nums), "n_rows": str(len(nums)),
             "fastq_gb": f"{g:.1f}"} for i, (nums, g) in enumerate(out, 1)]


def eic_accessions(ledger: list[dict], eic_rows: list[dict]) -> set:
    """Every accession the EIC side names as an experiment or a control."""
    s = OX.eic_sets(ledger, eic_rows)
    out = s["experiments"] | s["controls"] | s["build"]
    for r in ledger:
        out.update(B.split_acc(r.get("control_experiment", "")))
    return out


def check(rows: list[dict], selection: list[dict], ledger: list[dict], eic_rows: list[dict],
          out=print) -> int:
    expected = {}                                   # (experiment, kind) -> (track, priority)
    for kind in ("selected", "fallback"):
        for s in selection:
            x = s["experiment"] if kind == "selected" else s.get("fallback", "")
            if x:
                expected[x] = (s["eic_track"], PRIORITY[kind])
    lo = {PRIORITY[k]: FIRST_ROW[k] for k in FIRST_ROW}
    nsel = sum(1 for s in selection if s["experiment"])
    nfb = sum(1 for s in selection if s.get("fallback"))
    hi = {PRIORITY["selected"]: FIRST_ROW["selected"] + nsel - 1,
          PRIORITY["fallback"]: FIRST_ROW["fallback"] + nfb - 1}
    counts = Counter(r["experiment"] for r in rows)
    outside = []
    for r in rows:
        e = expected.get(r["experiment"])
        p = r["priority"]
        if (e is None or e != (r["eic_track"], p) or counts[r["experiment"]] > 1
                or p not in lo or not lo[p] <= int(r["row"]) <= hi[p]
                or r["unit"] != "pooled" or r["arm"] != "base"):
            outside.append(r["row"])
    missing = sorted(set(expected) - set(counts))
    out(f"rows outside the reviewed list: {len(outside)}" + (f"  {outside[:10]}" if outside else "")
        + (f"  (reviewed experiments without a row: {len(missing)})" if missing else ""))
    eic = eic_accessions(ledger, eic_rows)
    hit = sorted({r["experiment"] for r in rows if r["experiment"] in eic})
    out(f"experiments that are EIC experiments or EIC controls: {len(hit)}"
        + (f"  {hit[:10]}" if hit else ""))
    eic_nums = {r["row"] for r in eic_rows}
    shared = sorted((r["row"] for r in rows if r["row"] in eic_nums), key=int)
    out(f"row numbers shared with the EIC build list: {len(shared)}"
        + (f"  {shared[:10]}" if shared else ""))
    role = [r["row"] for r in rows if r["role"] != ROLE]
    out(f"rows with a role other than evalonly: {len(role)}" + (f"  {role[:10]}" if role else ""))
    sig = [r["row"] for r in rows if (r.get("control_experiment") or "").strip()
           or any(set(p) - set(B.BASE_PRODUCTS) or "signal" in "+".join(p)
                  for _, p in B.parse_tokens(r.get("expected_measurements", "")))]
    out(f"rows that request signal or name a control: {len(sig)}" + (f"  {sig[:10]}" if sig else ""))
    bad = len(outside) + len(missing) + len(hit) + len(shared) + len(role) + len(sig)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--selection", required=True, help="the reviewed list (OXs output)")
    b.add_argument("--catalog", required=True, help="D1 catalog directory (files.tsv)")
    b.add_argument("--out", required=True)
    b.add_argument("--batches", default="", help="also write the priority-5 batches here")
    c = sub.add_parser("check")
    c.add_argument("--build-list", required=True)
    c.add_argument("--selection", required=True)
    c.add_argument("--ledger", required=True)
    c.add_argument("--eic-build-list", required=True)
    a = ap.parse_args(argv)
    sel = B.read_tsv(a.selection)
    if a.cmd == "build":
        rows = build(sel, B.read_tsv(Path(a.catalog) / "files.tsv"))
        B.write_tsv(a.out, B.BUILD_LIST_COLUMNS, rows)
        if a.batches:
            B.write_tsv(a.batches, BATCH_COLUMNS, batches(rows))
        pc = Counter(r["priority"] for r in rows)
        print(f"rows: {len(rows)} (priority 5: {pc['5']}, priority 6: {pc['6']}); "
              f"native rows: {sum(1 for r in rows if r['pipeline'] == B.NATIVE_PIPELINE)}; "
              f"rows with a biological replicate mixing SE and PE FASTQs: "
              f"{sum(1 for r in rows if r['_mixed_rep'])}; "
              f"priority-5 fastq_gb: "
              f"{sum(float(r['fastq_gb']) for r in rows if r['priority'] == '5'):.1f}")
        return 0
    return check(B.read_tsv(a.build_list), sel, B.read_tsv(a.ledger),
                 B.read_tsv(a.eic_build_list))


if __name__ == "__main__":
    sys.exit(main())
