"""t112 counterfactual arms — every pinned constant of the plan, expanded into task TSVs.

One product per (track, arm, level). Each product differs from its track's base in exactly one
processing knob. This file holds the facts those products are built from — the 7 EIC tracks, their
3 controls, the arm table, and the knob value each (track, arm, level) takes — and nothing else.
Sibling tools (`bam_arm.py`, `fastq_arms.py`, `records.py`, `checks.py`) read its TSV or import it.

Values verified on Nibi 2026-09-17. A read count is the number of mapped primary records in the
nodup BAM (`samtools idxstats | awk '{m+=$3}'`), which is exactly the line count of the tagAlign
the pipeline's own `bam2ta` makes from that BAM. The `*.samstats.qc` files sitting beside these
BAMs describe a **different** run and must not be read for these counts. fraglen is `--fraglen` of
`calls["chip.macs2_signal_track"][0]["commandLine"]` in the track's `metadata.json`; read length is
sampled from the BAM.

A control read count is therefore the FULL control tagAlign: the base runs called `bam2ta_ctl` with
`--subsample 0` and `choose_ctl` returned `chosen_ctl_ta_subsample = [0]`, so MACS2 saw every read
of the control. `ctl_subsample_reads` and `macs2_ratio` are derived from that full count.

Two decisions are open with the PI and are **switches, not constants** here:

- D1, the DNase pipeline: `--dnase atac|none`. `none` drops every C12M02 row. Only `atac`
  (atac-seq-pipeline v2.2.3 in DNase mode) has an arm table; another pipeline is a re-plan.
- D2, the ratio arm (a one-token patched MACS2 signal script): `--ratio yes|no`.

Deliberately stdlib only: it runs on the Nibi login node, which has no numpy.

    python3 tools/t112/arms.py rows --route bam|fastq|all --dnase atac|none --ratio yes|no [--tracks T1,T2]
    python3 tools/t112/arms.py count --dnase atac|none --ratio yes|no
"""
from __future__ import annotations

import argparse
import json
import math
import sys

EIC = "/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline"

#: the treatment tagAlign MACS2 consumed in every ChIP base run (`--subsample 30000000`).
CHIP_SUBSAMPLE = 30000000


def _chip(track, cell, assay, tacc, reads, cacc, fraglen):
    return {
        "cell": cell, "assay": assay, "pipeline": "chip",
        "treat_acc": tacc,
        "treat_bam": f"{EIC}/results/{track}/filter_shard0_{tacc}.merged.srt.nodup.bam",
        "treat_reads": reads,
        "ctl_acc": cacc,
        "fraglen": fraglen, "fraglen_kind": "xcor",
        "read_length": 101, "run_type": "single-ended",
        "subsample": CHIP_SUBSAMPLE,
        "pval_bigwig": (f"{EIC}/results/{track}/macs2_signal_track_shard0_{tacc}.merged.srt.nodup"
                        f".30M_x_{cacc}.merged.srt.nodup.pval.signal.bigwig"),
    }


#: table order is the row order of every TSV.
TRACKS = {
    "C19M16": _chip("C19M16", "C19", "H3K27ac", "ENCFF254LWX", 176237758, "ENCFF433TZR", 180),
    "C40M17": _chip("C40M17", "C40", "H3K27me3", "ENCFF581NOL", 207716224, "ENCFF337JNL", 200),
    "C40M18": _chip("C40M18", "C40", "H3K36me3", "ENCFF443KLR", 239733719, "ENCFF337JNL", 210),
    "C07M20": _chip("C07M20", "C07", "H3K4me1", "ENCFF458MVX", 205681163, "ENCFF164WQW", 215),
    "C19M22": _chip("C19M22", "C19", "H3K4me3", "ENCFF748TKZ", 174409214, "ENCFF433TZR", 205),
    "C07M29": _chip("C07M29", "C07", "H3K9me3", "ENCFF777XCR", 217273310, "ENCFF164WQW", 200),
    "C12M02": {
        "cell": "C12", "assay": "DNase-seq",
        "pipeline": None,  # set by --dnase (Decision D1); rows carry the switch's value
        "treat_acc": "ENCFF211XVI",
        "treat_bam": f"{EIC}/results/C12M02/filter_shard0_ENCFF211XVI.merged.srt.nodup.no_chrM_MT.bam",
        "treat_reads": 134815660,
        "ctl_acc": None,
        "fraglen": 150, "fraglen_kind": "smooth_win",
        "read_length": 76, "run_type": "single-ended",
        "subsample": 50000000,
        "pval_bigwig": (f"{EIC}/results/C12M02/macs2_signal_track_shard0_ENCFF211XVI.merged.srt"
                        ".nodup.no_chrM_MT.50M.pval.signal.bigwig"),
    },
}

DNASE_TRACK = "C12M02"

CONTROLS = {
    "ENCFF433TZR": {"cell": "C19", "reads": 107039349, "read_length": 101, "run_type": "single-ended",
                    "bam": f"{EIC}/results/C19M16/filter_ctl_shard0_ENCFF433TZR.merged.srt.nodup.bam"},
    "ENCFF337JNL": {"cell": "C40", "reads": 150302589, "read_length": 101, "run_type": "single-ended",
                    "bam": f"{EIC}/results/C40M17/filter_ctl_shard0_ENCFF337JNL.merged.srt.nodup.bam"},
    "ENCFF164WQW": {"cell": "C07", "reads": 160028104, "read_length": 101, "run_type": "single-ended",
                    "bam": f"{EIC}/results/C07M20/filter_ctl_shard0_ENCFF164WQW.merged.srt.nodup.bam"},
}

#: arm 8 (ctlid/other): cyclic, every control used once as a foreign control.
CONTROL_ROTATION = {"C19": "ENCFF337JNL", "C40": "ENCFF164WQW", "C07": "ENCFF433TZR"}

#: (arm, route, knob, ((level, level parameter), ...)). Table order is row order.
#: The level parameter is the raw number; `knob_value` turns it into the value the knob takes.
CHIP_ARMS = (
    ("base", "bam", "none", (("base", None),)),
    ("depth", "bam", "treatment_reads", (("15M", 15000000), ("7.5M", 7500000), ("3.75M", 3750000))),
    ("pe", "fastq", "chip.paired_end", (("pe", True),)),
    ("dedup", "fastq", "chip.no_dup_removal", (("off", True),)),
    ("abproxy", "bam", "control_read_fraction", (("f0.5", 0.5), ("f0.9", 0.9))),
    ("crop", "fastq", "chip.crop_length", (("36", 36), ("50", 50))),
    ("mapq", "fastq", "chip.mapq_thresh", (("0", 0), ("10", 10))),
    ("ratio", "bam", "macs2_ratio", (("k0.5", 0.5), ("k2", 2))),
    ("ctlid", "bam", "control_accession", (("other", "other"), ("none", "none"))),
    ("ctldepth", "bam", "ctl_subsample_reads", (("q0.5", 0.5), ("q0.25", 0.25))),
    ("extsize", "bam", "fraglen", (("k0.5", 0.5), ("k2", 2))),
)

ARM_ORDER = tuple(a[0] for a in CHIP_ARMS)

#: per --dnase value. Not applicable under atac: abproxy, ratio, ctlid, ctldepth (no control) and
#: crop (atac.wdl v2.2.3 has no crop input).
DNASE_ARMS = {
    "atac": (
        ("base", "bam", "none", (("base", None),)),
        ("depth", "bam", "treatment_reads", (("15M", 15000000), ("7.5M", 7500000), ("3.75M", 3750000))),
        ("pe", "fastq", "atac.paired_end", (("pe", True),)),
        ("dedup", "fastq", "atac.no_dup_removal", (("off", True),)),
        ("mapq", "fastq", "atac.mapq_thresh", (("0", 0), ("10", 10))),
        ("extsize", "bam", "atac.smooth_win", (("k0.5", 0.5), ("k2", 2))),
    ),
}

HEADER = ("pid", "biosample", "track", "cell", "assay", "arm", "level", "route", "knob",
          "knob_value", "pipeline")


def round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def pid(track: str, arm: str, level: str) -> str:
    return f"{track}__{arm}__{level}"


def biosample(cell: str, arm: str, level: str) -> str:
    return f"CF_{cell}__{arm}__{level}"


def abproxy_n_ctl(fraction: float, total: int = CHIP_SUBSAMPLE) -> int:
    """Arm 4: control reads mixed into the treatment tagAlign; the total stays `total`."""
    return round_half_up(fraction * total)


def ratio_value(k: float, control_reads: int, total: int = CHIP_SUBSAMPLE) -> float:
    """Arm 7: MACS2 `--ratio`, k times the treatment/control depth ratio, 10 significant digits."""
    return float(f"{k * total / control_reads:.10g}")


def knob_value(track: str, arm: str, param):
    t = TRACKS[track]
    if arm == "abproxy":
        return param  # the fraction; n_ctl = abproxy_n_ctl(fraction)
    if arm == "ratio":
        return ratio_value(param, CONTROLS[t["ctl_acc"]]["reads"])
    if arm == "ctlid":
        return CONTROL_ROTATION[t["cell"]] if param == "other" else "none"
    if arm == "ctldepth":
        return round_half_up(param * CONTROLS[t["ctl_acc"]]["reads"])
    if arm == "extsize":
        return round_half_up(param * t["fraglen"])
    return param


def rows(route: str = "all", *, dnase: str, ratio: str, tracks=None) -> list[dict]:
    """Every (track, arm, level) row, sorted by track, arm, level in table order."""
    if route not in ("bam", "fastq", "all"):
        raise ValueError(f"route must be bam|fastq|all, not {route!r}")
    if dnase not in ("atac", "none"):
        raise ValueError(f"dnase must be atac|none, not {dnase!r}")
    if ratio not in ("yes", "no"):
        raise ValueError(f"ratio must be yes|no, not {ratio!r}")
    if tracks is not None:
        unknown = [x for x in tracks if x not in TRACKS]
        if unknown:
            raise ValueError(f"unknown tracks: {unknown}")
    out = []
    for track, t in TRACKS.items():
        if tracks is not None and track not in tracks:
            continue
        if track == DNASE_TRACK:
            if dnase == "none":
                continue
            arms, pipeline = DNASE_ARMS[dnase], dnase
        else:
            arms, pipeline = CHIP_ARMS, t["pipeline"]
        for arm, arm_route, knob, levels in arms:
            if arm == "ratio" and ratio == "no":
                continue
            if route != "all" and arm_route != route:
                continue
            for level, param in levels:
                out.append({
                    "pid": pid(track, arm, level),
                    "biosample": biosample(t["cell"], arm, level),
                    "track": track, "cell": t["cell"], "assay": t["assay"],
                    "arm": arm, "level": level, "route": arm_route, "knob": knob,
                    "knob_value": knob_value(track, arm, param),
                    "pipeline": pipeline,
                })
    return out


def to_tsv(rs: list[dict]) -> str:
    lines = ["\t".join(HEADER)]
    for r in rs:
        lines.append("\t".join(json.dumps(r[k]) if k == "knob_value" else str(r[k]) for k in HEADER))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_rows = sub.add_parser("rows", help="write the arm rows as TSV to stdout")
    p_rows.add_argument("--route", choices=("bam", "fastq", "all"), required=True)
    p_count = sub.add_parser("count", help="print the row count of `rows --route all`")
    for p in (p_rows, p_count):
        p.add_argument("--dnase", choices=("atac", "none"), required=True, help="Decision D1")
        p.add_argument("--ratio", choices=("yes", "no"), required=True, help="Decision D2")
    p_rows.add_argument("--tracks", default=None, help="comma-separated subset, e.g. C19M16,C12M02")
    args = ap.parse_args(argv)

    if args.cmd == "count":
        print(len(rows("all", dnase=args.dnase, ratio=args.ratio)))
        return 0
    tracks = args.tracks.split(",") if args.tracks else None
    try:
        rs = rows(args.route, dnase=args.dnase, ratio=args.ratio, tracks=tracks)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    sys.stdout.write(to_tsv(rs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
