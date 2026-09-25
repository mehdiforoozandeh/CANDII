"""t118 — the per-product covariate table: one full knob vector for every t112 product.

The t112 MANIFEST records depth, read length, run type and fragment length for every product,
but dedup, MAPQ, control fraction, MACS2 ratio, control depth and extsize only on the arm that
changes them. This tool fills in the base value of every knob and names, per value, where it
came from (`src_*` columns). Where the base value is a pipeline default, the WDL line is the one
at the pinned release (chip-seq-pipeline2 v2.2.2 `chip.wdl`, atac-seq-pipeline v2.2.3
`atac.wdl`, both read from the ENCODE-DCC tags on 2026-09-25):

    chip.wdl v2.2.2  L177 String dup_marker = 'picard'   L178 Boolean no_dup_removal = false
                     L179 Int mapq_thresh = 30
    atac.wdl v2.2.3  L181 Int multimapping = 4           L182 String dup_marker = 'picard'
                     L183 Boolean no_dup_removal = false L184 Int mapq_thresh = 30

The t112 FASTQ arms change exactly one input key of the base input JSON
(`tools/t112/fastq_arms.py::build_json`), so every other key stays at what the base ran with.

Columns are pinned (the t118 ladder's encoder reads them by name). Output is deterministic: rows
sorted by pid, floats written with `repr`, ints as ints, tab-separated, LF.

It also re-verifies, from the manifest's `counts25_md5`, that the only products whose counts are
byte-identical to their track's base are the 50 p-only arm products (ratio, ctlid, ctldepth,
extsize) and the 2 DNase MAPQ arms (a t112 defect: `multimapping = 4` voids the MAPQ cut; t119
rebuilds them). Those two get `usable = 0`.

Stdlib only, so it runs on a Nibi login node.

    python tools/t118/build_covariates.py <MANIFEST.tsv> <covariates_by_pid.json | products_dir> <out.tsv>

The products_dir form reads `<dir>/<pid>/covariates.json` for every manifest pid.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

COLUMNS = (
    "pid", "track", "assay", "arm", "level", "usable",
    "depth_reads", "depth_log2", "read_length", "run_type_pe", "dedup_on", "mapq_thresh",
    "has_control", "control_fraction", "ratio_k", "ctl_identity", "control_reads",
    "control_depth_log2", "extsize_k", "fraglen_bp",
    "src_depth", "src_read_length", "src_run_type", "src_dedup", "src_mapq",
    "src_control_fraction", "src_ratio_k", "src_ctl_identity", "src_control_depth",
    "src_extsize_k", "src_fraglen",
)

#: pipeline_repo -> (wdl prefix, pinned release, wdl line of no_dup_removal, of mapq_thresh).
PIPELINES = {
    "ENCODE-DCC/chip-seq-pipeline2": ("chip", "v2.2.2", 178, 179),
    "ENCODE-DCC/atac-seq-pipeline": ("atac", "v2.2.3", 183, 184),
}
ATAC_MULTIMAPPING_LINE = 181

#: byte-identical to the DNase base (t112 defect); excluded from training and scoring until t119.
UNUSABLE = frozenset({"C12M02__mapq__0", "C12M02__mapq__10"})

#: arms that change only -log10 p: their counts equal the base's by construction.
P_ONLY_ARMS = frozenset({"ratio", "ctlid", "ctldepth", "extsize"})

MAPQ_BASE = 30
NO_CONTROL = "none: no control"


def _k(level: str) -> float:
    """`k0.5` -> 0.5, `k2` -> 2.0 (the ratio and extsize levels)."""
    m = re.fullmatch(r"k(\d+(?:\.\d+)?)", level)
    if not m:
        raise SystemExit(f"level {level!r} is not k<number>")
    return float(m.group(1))


def _fmt(v) -> str:
    if isinstance(v, bool):
        raise TypeError("write flags as 0/1 ints")
    if isinstance(v, float):
        return repr(v)
    return str(v)


def read_manifest(path) -> list[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    for r in rows:
        r["knob_value"] = json.loads(r["knob_value"]) if r["knob_value"] != "" else None
    return rows


def read_covariates(src, pids) -> dict:
    src = Path(src)
    if src.is_dir():
        return {p: json.loads((src / p / "covariates.json").read_text()) for p in pids}
    return json.loads(src.read_text())


def check_identical_counts(rows: list[dict]) -> set[str]:
    """The pids whose counts25_md5 equals their track base's; must be the 52 expected."""
    base = {r["track"]: r["counts25_md5"] for r in rows if r["arm"] == "base"}
    same = {r["pid"] for r in rows if r["arm"] != "base" and r["counts25_md5"] == base[r["track"]]}
    expected = {r["pid"] for r in rows if r["arm"] in P_ONLY_ARMS} | UNUSABLE
    if same != expected or len(same) != 52:
        raise SystemExit(f"counts-identical set changed: extra {sorted(same - expected)}, "
                         f"missing {sorted(expected - same)}, n = {len(same)}")
    return same


def row_for(m: dict, cov: dict) -> dict:
    pid, arm, level, knob, kv = m["pid"], m["arm"], m["level"], m["knob"], m["knob_value"]
    for key in ("pid", "track", "assay", "arm", "level", "depth", "read_length", "run_type",
                "fraglen", "knob"):
        if str(cov[key]) != m[key]:
            raise SystemExit(f"{pid}: manifest {key}={m[key]!r} but covariates.json {cov[key]!r}")
    if cov["knob_value"] != kv:
        raise SystemExit(f"{pid}: manifest knob_value={kv!r} but covariates.json {cov['knob_value']!r}")
    prefix, release, dedup_line, mapq_line = PIPELINES[m["pipeline_repo"]]
    if m["pipeline_release"] != release:
        raise SystemExit(f"{pid}: pipeline {m['pipeline_release']} is not the pinned {release}; "
                         "the WDL line numbers in the src_* strings would be wrong")
    wdl = f"{prefix}.wdl {release} default"

    ctl = cov["control"]
    has_control = ctl is not None
    if has_control != (m["control_source"] != ""):
        raise SystemExit(f"{pid}: control presence differs between manifest and covariates.json")
    if has_control and (ctl["source"], ctl["accession"]) != (m["control_source"], m["control_accession"]):
        raise SystemExit(f"{pid}: control differs between manifest and covariates.json")

    depth = int(m["depth"])
    if cov["log2_depth"] != math.log2(depth):
        raise SystemExit(f"{pid}: covariates.json log2_depth disagrees with log2(depth)")

    if arm == "dedup":
        if kv is not True:
            raise SystemExit(f"{pid}: dedup arm knob_value {kv!r} is not true")
        dedup_on, src_dedup = 0, f"manifest:knob_value ({knob})"
    else:
        dedup_on, src_dedup = 1, f"{wdl} L{dedup_line} no_dup_removal = false (picard)"

    void = (f"; multimapping = 4 (atac.wdl {release} L{ATAC_MULTIMAPPING_LINE}) voided the cut "
            "on the t112 mapq arms (t119)") if prefix == "atac" else ""
    if arm == "mapq":
        mapq, src_mapq = int(kv), f"manifest:knob_value ({knob}){void}"
    else:
        mapq, src_mapq = MAPQ_BASE, f"{wdl} L{mapq_line} mapq_thresh = 30{void}"

    if arm == "abproxy":
        cfrac, src_cfrac = float(kv), f"manifest:knob_value ({knob})"
    elif has_control:
        cfrac, src_cfrac = 0.0, "tools/t112/arms.py:abproxy_n_ctl (off this arm no control read is mixed in)"
    else:
        cfrac, src_cfrac = 0.0, NO_CONTROL

    if arm == "ratio":
        ratio, src_ratio = _k(level), "tools/t112/bam_arm.py:ratio_for (k from manifest:level)"
    elif has_control:
        ratio, src_ratio = 1.0, "tools/t112/bam_arm.py:ratio_for (k = 1 is MACS2's own scaling)"
    else:
        ratio, src_ratio = 0.0, NO_CONTROL

    if has_control:
        ident, src_ident = ctl["source"], "covariates.json:control.source"
        creads, src_creads = int(ctl["reads"]), "covariates.json:control.reads"
        clog2 = math.log2(creads)
    else:
        ident, src_ident = "none", NO_CONTROL
        creads, src_creads, clog2 = 0, NO_CONTROL, 0.0

    if arm == "extsize":
        ext, src_ext = _k(level), f"manifest:level ({knob} = k x base fraglen)"
    else:
        ext, src_ext = 1.0, "tools/t112/arms.py:knob_value (k = 1: the base fraglen)"

    return {
        "pid": pid, "track": m["track"], "assay": m["assay"], "arm": arm, "level": level,
        "usable": 0 if pid in UNUSABLE else 1,
        "depth_reads": depth, "depth_log2": math.log2(depth),
        "read_length": int(m["read_length"]),
        "run_type_pe": 1 if m["run_type"] == "paired-ended" else 0,
        "dedup_on": dedup_on, "mapq_thresh": mapq,
        "has_control": 1 if has_control else 0, "control_fraction": cfrac, "ratio_k": ratio,
        "ctl_identity": ident, "control_reads": creads, "control_depth_log2": clog2,
        "extsize_k": ext, "fraglen_bp": int(m["fraglen"]),
        "src_depth": "manifest:depth", "src_read_length": "manifest:read_length",
        "src_run_type": "manifest:run_type", "src_dedup": src_dedup, "src_mapq": src_mapq,
        "src_control_fraction": src_cfrac, "src_ratio_k": src_ratio,
        "src_ctl_identity": src_ident, "src_control_depth": src_creads,
        "src_extsize_k": src_ext, "src_fraglen": "manifest:fraglen",
    }


def build(manifest, covariates) -> list[dict]:
    rows = read_manifest(manifest)
    pids = [r["pid"] for r in rows]
    if len(set(pids)) != len(pids):
        raise SystemExit("duplicate pid in manifest")
    covs = read_covariates(covariates, pids)
    if set(covs) != set(pids):
        raise SystemExit(f"covariates pids differ from manifest: "
                         f"{sorted(set(covs) ^ set(pids))}")
    check_identical_counts(rows)
    return [row_for(m, covs[m["pid"]]) for m in sorted(rows, key=lambda r: r["pid"])]


def render(table: list[dict]) -> str:
    lines = ["\t".join(COLUMNS)]
    lines += ["\t".join(_fmt(r[c]) for c in COLUMNS) for r in table]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print("usage: build_covariates.py <MANIFEST.tsv> <covariates_by_pid.json | products_dir> "
              "<out.tsv>", file=sys.stderr)
        return 2
    manifest, covariates, out = argv
    text = render(build(manifest, covariates))
    with open(out, "w", newline="\n") as fh:
        fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
