#!/usr/bin/env python3
"""EIC build list in priority order, and the withheld (never-trained) covariate combinations
(t170, chunk D4; decisions D5, D7, D8; PI ruling 8).

    build_list.py probe --ledger L --catalog DIR --out PROBE.tsv
    build_list.py build --ledger L --catalog DIR --probe PROBE.tsv --out BUILD_LIST.tsv
                        --withheld-out WITHHELD.tsv [--summary-dir DIR]
    build_list.py check --build-list BUILD_LIST.tsv --ledger L --withheld WITHHELD.tsv

Inputs. `L` is the D3 ledger (`ONION/eic/accession_ledger.tsv`). `DIR` is the D1 ENCODE catalog
(`ONION/catalog/encode/`, its `files.tsv`). `probe` sends one HTTP HEAD per FASTQ (and per GRCh38
"redacted unfiltered alignments" BAM) of every ledger experiment and of every ENCODE possible
control a ledger ChIP experiment names; a file enters a build row only when the portal served it
(HTTP 200). Restricted FASTQs answer 403.

Experiments with no served FASTQ (restricted Roadmap tissues: whole cells) are built from ENCODE's
own public "redacted unfiltered alignments" GRCh38 BAMs (read names, flags, MAPQ and positions
kept; sequence replaced by N): `layer = native`, `pipeline = native-redacted-bam`, and the row's
`fastq_files` column then holds the BAM accessions (one per biological replicate) and `fastq_gb`
their size. No FASTQ kit accepts such a row (`fastq_uniform.pipeline_key` refuses the pipeline
name); `align_views.py` runs the pipeline's own filter step on the BAM. No FASTQ-level arm exists
for them. An experiment with neither is excluded with its reason.

Which experiments (D8 and the foreman's rulings of 2026-11-11):
  * every ledger experiment with `match = 1`, under its ledger role. A track is built from the
    PUBLISHED experiment's own reads (`experiment`), never from the store file's experiment.
  * controls come from the ENCODE record: the ledger's `encode_possible_controls` of each ChIP
    experiment, every one of them. A control's role is `train` when it controls at least one
    training experiment, else `val` when it controls a validation experiment, else `blind`.
  * blind tracks (the 7 testbed tracks among them) are built only in priority 4, as blind.

Priorities ("Early start" of the plan):
  1  validation experiments and the controls of validation experiments: pooled base (with
     per-replicate views, halves, signal tracks and the alignment-level arms), then the
     FASTQ-level arms. Order: base rows of the scorable assays first, assay by assay (the assay
     whose validation set is smallest in GB first, so a complete scorable assay comes earliest),
     then base rows of the other assays, then the FASTQ-level arms.
  2  training experiments and the controls of training experiments: pooled base and the
     alignment-level arms. Order: cell by cell, cells with the most scorable validation tracks
     first (a validation target's inputs are its cell's training tracks), controls first.
  3  FASTQ-level arms on training experiments: `se` on EVERY paired-end training experiment (the
     blind set is 100% paired-end), `crop36` and `crop50` on up to CROP_PER_ASSAY experiments per
     assay, chosen greedily to cover new metadata strata (lab, platform, read-length bin, depth
     tercile within assay, run type).
  4  blind experiments and their controls (base, halves, signal, alignment-level and FASTQ-level
     arms), then per-library units of train and validation experiments (bio reps with 2 or more
     libraries), then per-lane units (libraries with 2 or more runs).

Products (`expected_measurements`): `;`-separated `<measurement_id>:<product>+<product>...`.
`measurement_id = {experiment}.{unit}.uniform.{arm}`. Halves `pr1`, `pr2` of EVERY arm of a parent
`{experiment}.{unit}` come from ONE read-level split of the parent (by read name, as
`align_views.py` does), so `<arm u>.pr1` and `<arm v>.pr2` share no read: scored correction pairs
read the source from pr1 and the target from pr2 (PI ruling 6; addendum). Validation and blind base
rows also request `signal`, `pr1_signal`, `pr2_signal`.

Withheld design: `WITHHELD_DESIGN`, written per assay class only where both views exist in
training of that class. `check` verifies the three gate lines of the plan and the rule lines.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import math
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROLES = ("train", "val", "blind")
CONTROL_CLASS = "control"
HISTONE = ("broad_histone", "narrow_histone")
PIPELINE = {
    "broad_histone": "chip-seq-pipeline2 v2.2.2", "narrow_histone": "chip-seq-pipeline2 v2.2.2",
    "control": "chip-seq-pipeline2 v2.2.2", "accessibility_atac": "atac-seq-pipeline v2.2.3",
    "accessibility_dnase": "dnase-seq-pipeline v3.0.0",
}
BUILD_LIST_COLUMNS = ("row", "priority", "experiment", "eic_track", "role", "assay", "assay_class",
                      "unit", "arm", "fastq_files", "fastq_gb", "control_experiment", "pipeline",
                      "keep_bam", "expected_measurements")
WITHHELD_COLUMNS = ("source_knob", "source_value", "target_knob", "target_value", "assay_class")
PROBE_COLUMNS = ("file", "experiment", "status", "http", "content_length", "catalog_size")
NATIVE_PIPELINE = "native-redacted-bam"
NATIVE_OUTPUT = "redacted unfiltered alignments"


def is_native_bam(f: dict) -> bool:
    return (f.get("file_format") == "bam" and f.get("output_type") == NATIVE_OUTPUT
            and f.get("assembly") == "GRCh38")

#: arm name -> (arm_knob, arm_value) as the kits write them into measurements.tsv:
#: `se` -> ("se", "1") is `fastq_uniform.ARMS`; `nodedup` -> ("nodedup", "") and `mapq0` ->
#: ("mapq", "0") are `align_views.arm_fields`. `minicandi.dataset.Withheld` matches on these.
ARM_VIEW = {
    "base": ("base", ""), "se": ("se", "1"), "crop36": ("crop", "36"), "crop50": ("crop", "50"),
    "depth0.50": ("depth", "0.50"), "depth0.25": ("depth", "0.25"), "depth0.125": ("depth", "0.125"),
    "nodedup": ("nodedup", ""), "mapq0": ("mapq", "0"),
}
FASTQ_ARMS = ("se", "crop36", "crop50")
CROPS = {"crop36": 36, "crop50": 50}
DEPTH_ARMS = ("depth0.50", "depth0.25", "depth0.125")
ALIGN_ARMS = DEPTH_ARMS + ("nodedup", "mapq0")
#: views whose pairs are pure depth thinning of one read set (subset_pair = 1): never counted.
DEPTH_ONLY = {"base"} | set(DEPTH_ARMS)
CROP_PER_ASSAY = 3
#: (source arm, target arm, assay classes). Combinations, never whole knobs: every arm keeps
#: trained pairs with base and the other arms on both sides. See results/t170/withheld_design.md.
WITHHELD_DESIGN = (
    ("nodedup", "mapq0", HISTONE + ("accessibility_dnase",)),
    ("mapq0", "nodedup", HISTONE + ("accessibility_dnase",)),
    ("crop36", "crop50", HISTONE + ("accessibility_atac", "accessibility_dnase")),
    ("crop50", "crop36", HISTONE + ("accessibility_atac", "accessibility_dnase")),
    ("se", "crop50", HISTONE + ("accessibility_atac", "accessibility_dnase")),
    ("crop50", "se", HISTONE + ("accessibility_atac", "accessibility_dnase")),
)
BASE_PRODUCTS = ("counts", "pr1", "pr2")
SIGNAL_PRODUCTS = ("signal", "pr1_signal", "pr2_signal")


# --------------------------------------------------------------------------------------------- io

def read_tsv(path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path, columns, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t", lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    tmp.replace(path)


def split_acc(s: str) -> list[str]:
    return [x for x in (s or "").replace(";", ",").replace(" ", ",").split(",") if x]


def _same_value(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def view_arm(knob: str, value: str) -> str | None:
    """The arm name of a (knob, value) pair, or None when no arm of ARM_VIEW has it."""
    knob = knob or "base"
    for arm, (k, v) in ARM_VIEW.items():
        if k == knob and _same_value(v, value or ""):
            return arm
    return None


def is_subset_pair(u: str, v: str) -> bool:
    """True when two views differ only by read-level depth thinning (subset_pair = 1)."""
    return u in DEPTH_ONLY and v in DEPTH_ONLY


# ----------------------------------------------------------------------------- ledger and catalog

def ledger_sets(ledger: list[dict]) -> dict:
    """EIC experiments (match = 1, by role) and the ENCODE possible controls of ChIP rows."""
    eic = {}
    for r in ledger:
        if r["assay_class"] == CONTROL_CLASS or r.get("match") != "1":
            continue
        eic[r["experiment"]] = r
    controls = defaultdict(set)          # control experiment -> roles of the experiments it controls
    controlled = defaultdict(list)       # control experiment -> ChIP experiments it controls
    for x, r in eic.items():
        if r["assay_class"] in HISTONE:
            for c in split_acc(r.get("encode_possible_controls", "")):
                controls[c].add(r["role"])
                controlled[c].append(x)
    ctl_tracks = defaultdict(set)
    for r in ledger:
        if r["assay_class"] == CONTROL_CLASS:
            ctl_tracks[r["experiment"]].add(r["eic_track"])
    return {"eic": eic, "controls": controls, "controlled": controlled, "ctl_tracks": ctl_tracks,
            "store_file_experiments": {r.get("file_experiment", "") for r in ledger} - {""}}


def control_role(roles: set) -> str:
    return "train" if "train" in roles else ("val" if "val" in roles else "blind")


def wanted_experiments(ledger) -> set:
    s = ledger_sets(ledger)
    return set(s["eic"]) | set(s["controls"])


def native_bams(files: list[dict], probe: dict | None) -> list[dict]:
    """Served GRCh38 redacted unfiltered BAMs, one per biological replicate: released ones, or the
    archived ones when none is released (the rule the kits apply to FASTQs)."""
    out = {}
    cand = [f for f in files if is_native_bam(f)]
    cand = [f for f in cand if f.get("status") == "released"] or \
           [f for f in cand if f.get("status") == "archived"]
    for f in sorted(cand, key=lambda f: f["file"]):
        if probe is not None and str((probe.get(f["file"]) or {}).get("http")) != "200":
            continue
        out.setdefault(f.get("bio_reps", ""), f)
    return [out[k] for k in sorted(out)]


def read_probe(path) -> dict:
    return {r["file"]: r for r in read_tsv(path)}


def available_fastqs(files: list[dict], probe: dict | None) -> tuple[list[dict], list[dict]]:
    """(kept, dropped) FASTQs of one experiment. Released FASTQs only, unless the experiment has
    none (an archived experiment: its archived FASTQs, as the kits do). A FASTQ the portal did not
    serve is dropped, and so is its mate."""
    fq = [f for f in files if f.get("file_format", "fastq") == "fastq"]
    rel = [f for f in fq if f.get("status") == "released"] or \
          [f for f in fq if f.get("status") == "archived"]

    def served(f):
        if probe is None:
            return bool(f.get("file_size"))
        p = probe.get(f["file"])
        return p is not None and str(p.get("http")) == "200"

    bad = {f["file"] for f in rel if not served(f)}
    accs = {f["file"] for f in rel}
    for f in rel:
        mate = f.get("paired_with", "")
        if f.get("run_type") == "paired-ended" and (mate in bad or mate not in accs):
            bad.add(f["file"])
            if mate:
                bad.add(mate)
    kept = [f for f in rel if f["file"] not in bad]
    dropped = [f for f in rel if f["file"] in bad]
    return sorted(kept, key=lambda f: f["file"]), dropped


def gb(files) -> float:
    return sum(int(f.get("file_size") or 0) for f in files) / 1e9


def tech_of(f) -> int:
    t = f.get("tech_reps", "")
    return int(t.split("_")[1]) if "_" in t else 1


def bio_of(f) -> int:
    return int(split_acc(f.get("bio_reps", "1"))[0])


def facts(kept: list[dict]) -> dict:
    """Run type, read lengths, replicate and library structure of the kept FASTQs."""
    r1 = [f for f in kept if f.get("paired_end", "") in ("", "1")]
    pe = any(f.get("run_type") == "paired-ended" for f in kept)
    lengths = [int(f["read_length"]) for f in kept if f.get("read_length")]
    reps = defaultdict(lambda: defaultdict(list))   # bio -> tech -> R1 files
    for f in r1:
        reps[bio_of(f)][tech_of(f)].append(f)
    mixed_rep = sorted(b for b in reps
                       if len({f.get("run_type") for t in reps[b].values() for f in t}) > 1)
    return {"pe": pe, "min_rl": min(lengths) if lengths else 0, "reps": reps,
            "mixed_rep": mixed_rep, "platforms": sorted({f.get("platform", "") for f in kept}),
            "run_types": sorted({f.get("run_type", "") for f in kept})}


def mates(f, by_acc) -> list[dict]:
    m = by_acc.get(f.get("paired_with", "")) if f.get("run_type") == "paired-ended" else None
    return [f] + ([m] if m else [])


# ---------------------------------------------------------------------------------- arms, products

def fastq_arms_applicable(fa: dict) -> list[str]:
    out = ["se"] if fa["pe"] else []
    return out + [a for a, n in CROPS.items() if fa["min_rl"] > n]


def align_arms(assay_class: str) -> tuple[str, ...]:
    # align_views refuses mapq0 under --multimapping > 0 (the ATAC default 4).
    if assay_class == "accessibility_atac":
        return DEPTH_ARMS + ("nodedup",)
    return ALIGN_ARMS


def token(mid: str, products) -> str:
    return f"{mid}:{'+'.join(products)}"


def parse_tokens(s: str) -> list[tuple[str, tuple[str, ...]]]:
    out = []
    for t in (s or "").split(";"):
        if not t:
            continue
        mid, _, prods = t.partition(":")
        out.append((mid, tuple(p for p in prods.split("+") if p)))
    return out


def mid_parts(mid: str) -> tuple[str, str, str, str]:
    """(experiment, unit, layer, arm); the arm may itself hold a dot (depth0.50)."""
    exp, unit, layer, arm = mid.split(".", 3)
    return exp, unit, layer, arm


# ------------------------------------------------------------------------------------------ build

def strata_key(rec: dict, fa: dict, tercile: str) -> tuple:
    rl = fa["min_rl"]
    rlbin = "<=36" if rl <= 36 else ("37-75" if rl <= 75 else ">=76")
    return (("lab", rec.get("lab", "")), ("platform", ",".join(fa["platforms"])),
            ("rl", rlbin), ("depth", tercile), ("rt", "PE" if fa["pe"] else "SE"))


def depth_terciles(eic: dict) -> dict:
    out = {}
    by = defaultdict(list)
    for x, r in eic.items():
        if r["role"] == "train":
            try:
                by[r["assay"]].append((float(r.get("depth") or 0), x))
            except ValueError:
                by[r["assay"]].append((0.0, x))
    for assay, v in by.items():
        v.sort()
        n = len(v)
        for i, (_, x) in enumerate(v):
            out[x] = str(min(2, (3 * i) // max(n, 1)))
    return out


def scorable_assays(eic: dict) -> set:
    """Plan pinned interface D, imputation: >= 3 validation tracks from >= 3 cells and >= 5
    training tracks (halves are built for every validation track here)."""
    val, cells, train = Counter(), defaultdict(set), Counter()
    for r in eic.values():
        if r["role"] == "val":
            val[r["assay"]] += 1
            cells[r["assay"]].add(r["cell"])
        elif r["role"] == "train":
            train[r["assay"]] += 1
    return {a for a in val if val[a] >= 3 and len(cells[a]) >= 3 and train[a] >= 5}


def build(ledger: list[dict], catalog_files: list[dict], probe: dict | None) -> dict:
    S = ledger_sets(ledger)
    eic, controls = S["eic"], S["controls"]
    by_exp = defaultdict(list)
    for f in catalog_files:
        if f.get("file_format", "fastq") == "fastq" or is_native_bam(f):
            by_exp[f["experiment"]].append(f)
    scorable = scorable_assays(eic)
    terc = depth_terciles(eic)

    # one record per experiment to build
    exps = {}
    excluded = []
    for x, r in eic.items():
        exps[x] = {"experiment": x, "eic_track": r["eic_track"], "role": r["role"],
                   "assay": r["assay"], "assay_class": r["assay_class"], "cell": r["cell"],
                   "lab": r.get("lab", ""), "control_experiment":
                   ";".join(split_acc(r.get("encode_possible_controls", "")))
                   if r["assay_class"] in HISTONE else "",
                   "in_testbed": r.get("in_testbed", "0")}
    for c, roles in controls.items():
        cells = sorted({eic[x]["cell"] for x in S["controlled"][c]})
        exps[c] = {"experiment": c, "eic_track": ";".join(sorted(S["ctl_tracks"].get(c, ()))),
                   "role": control_role(roles), "assay": "chipseq-control",
                   "assay_class": CONTROL_CLASS, "cell": ";".join(cells), "lab": "",
                   "control_experiment": "", "controls_roles": roles, "in_testbed": "0"}
    dropped_files = []
    for x, e in list(exps.items()):
        kept, dropped = available_fastqs(by_exp.get(x, []), probe)
        e["kept"], e["by_acc"] = kept, {f["file"]: f for f in kept}
        e["gb"] = gb(kept)
        e["native"] = []
        for f in dropped:
            dropped_files.append({"experiment": x, "file": f["file"], "eic_track": e["eic_track"],
                                  "role": e["role"], "reason": "not served by the portal "
                                  "(restricted or no file), or its mate was not"})
        if not kept:
            bams = native_bams(by_exp.get(x, []), probe)
            if not bams:
                excluded.append({"experiment": x, "eic_track": e["eic_track"], "role": e["role"],
                                 "assay": e["assay"], "reason": "no FASTQ and no redacted "
                                 "unfiltered GRCh38 BAM served by the portal"})
                del exps[x]
                continue
            e["native"], e["gb"] = bams, gb(bams)
            reps = defaultdict(lambda: defaultdict(list))
            for f in bams:
                reps[bio_of(f)][1].append(f)
            e["facts"] = {"pe": False, "min_rl": 0, "reps": reps, "mixed_rep": [],
                          "platforms": [], "run_types": []}
            continue
        e["facts"] = facts(kept)
    for x, e in exps.items():
        if e["assay_class"] != CONTROL_CLASS:
            e["strata"] = strata_key(e, e["facts"], terc.get(x, ""))

    # FASTQ-level arms on training experiments (priority 3)
    p3 = defaultdict(list)                     # experiment -> arms
    train_x = sorted((x for x, e in exps.items()
                      if e["role"] == "train" and e["assay_class"] != CONTROL_CLASS),
                     key=lambda x: (exps[x]["gb"], x))
    for x in train_x:
        if exps[x]["facts"]["pe"]:
            p3[x].append("se")
    by_assay = defaultdict(list)
    for x in train_x:
        by_assay[exps[x]["assay"]].append(x)
    for assay, xs in sorted(by_assay.items()):
        for arm, n in CROPS.items():
            elig = [x for x in xs if exps[x]["facts"]["min_rl"] > n]
            seen, chosen = set(), []
            while elig and len(chosen) < CROP_PER_ASSAY:
                best = max(elig, key=lambda x: (len(set(exps[x]["strata"]) - seen),
                                                -exps[x]["gb"], x))
                chosen.append(best)
                seen |= set(exps[best]["strata"])
                elig.remove(best)
            for x in chosen:
                p3[x].append(arm)

    # views trained per class (eic_v1 = priorities 1 to 3, role train)
    train_views = defaultdict(set)
    for x, e in exps.items():
        if e["role"] == "train":
            train_views[e["assay_class"]] |= {"base", *align_arms(e["assay_class"]), *p3.get(x, [])}

    def eval_arms(e):          # FASTQ arms for a validation or blind experiment
        if e.get("native"):
            return []
        return [a for a in fastq_arms_applicable(e["facts"]) if a in train_views[e["assay_class"]]]

    withheld = []
    for src, tgt, classes in WITHHELD_DESIGN:
        for c in classes:
            if src in train_views[c] and tgt in train_views[c]:
                (sk, sv), (tk, tv) = ARM_VIEW[src], ARM_VIEW[tgt]
                withheld.append({"source_knob": sk, "source_value": sv, "target_knob": tk,
                                 "target_value": tv, "assay_class": c})

    # rows
    rows = []

    def add(priority, e, unit, arm, files, mids, keep_bam=0, files_gb=None):
        native = bool(e.get("native"))
        rows.append({
            "priority": str(priority), "experiment": e["experiment"], "eic_track": e["eic_track"],
            "role": e["role"], "assay": e["assay"], "assay_class": e["assay_class"],
            "unit": unit, "arm": arm, "fastq_files": ";".join(f["file"] for f in files),
            "fastq_gb": f"{(gb(files) if files_gb is None else files_gb):.3f}",
            "control_experiment": e["control_experiment"],
            "pipeline": NATIVE_PIPELINE if native else PIPELINE[e["assay_class"]],
            "keep_bam": str(keep_bam),
            "expected_measurements": ";".join(mids)})

    def base_row(priority, e, keep_bam):
        x, cls = e["experiment"], e["assay_class"]
        layer = "native" if e.get("native") else "uniform"
        scored = e["role"] in ("val", "blind") and cls != CONTROL_CLASS
        mids = [token(f"{x}.pooled.{layer}.base",
                      BASE_PRODUCTS + (SIGNAL_PRODUCTS if scored else ()))]
        bios = sorted(e["facts"]["reps"])
        if len(bios) > 1:
            mids += [token(f"{x}.br{b}.{layer}.base", BASE_PRODUCTS) for b in bios]
        mids += [token(f"{x}.pooled.{layer}.{a}", BASE_PRODUCTS) for a in align_arms(cls)]
        add(priority, e, "pooled", "base", e.get("native") or e["kept"], mids, keep_bam)

    def arm_row(priority, e, arm):
        x = e["experiment"]
        add(priority, e, "pooled", arm, e["kept"],
            [token(f"{x}.pooled.uniform.{arm}", BASE_PRODUCTS)])

    # priority 1: validation experiments and their controls
    val_x = [x for x, e in exps.items() if e["role"] == "val" and e["assay_class"] != CONTROL_CLASS]
    val_ctl = {c for c, e in exps.items() if e["assay_class"] == CONTROL_CLASS
               and "val" in e.get("controls_roles", ())}

    def ctl_of(x):
        return [c for c in split_acc(exps[x]["control_experiment"]) if c in exps]

    def ctl_gb(xs):
        return sum(exps[c]["gb"] for c in {c for x in xs for c in ctl_of(x)})

    assay_val = defaultdict(list)
    for x in val_x:
        assay_val[exps[x]["assay"]].append(x)
    order = sorted(assay_val, key=lambda a: (a not in scorable,
                                             sum(exps[x]["gb"] for x in assay_val[a])
                                             + ctl_gb(assay_val[a]), a))
    placed = set()
    for a in order:
        for x in sorted(assay_val[a], key=lambda x: (exps[x]["gb"], x)):
            for c in ctl_of(x):
                if c in val_ctl and c not in placed:
                    base_row(1, exps[c], 0)
                    placed.add(c)
            base_row(1, exps[x], 1)
            placed.add(x)
    for c in sorted(val_ctl - placed):
        base_row(1, exps[c], 0)
        placed.add(c)
    for a in order:
        for x in sorted(assay_val[a], key=lambda x: (exps[x]["gb"], x)):
            for arm in eval_arms(exps[x]):
                arm_row(1, exps[x], arm)

    # priority 2: training experiments and controls of training experiments
    nscor = Counter(exps[x]["cell"] for x in val_x if exps[x]["assay"] in scorable)
    tr_by_cell = defaultdict(list)
    for x in train_x:
        tr_by_cell[exps[x]["cell"]].append(x)
    cells = sorted(tr_by_cell, key=lambda c: (-nscor[c], sum(exps[x]["gb"] for x in tr_by_cell[c]), c))
    for cell in cells:
        xs = sorted(tr_by_cell[cell], key=lambda x: (exps[x]["assay"] not in scorable, exps[x]["gb"], x))
        for x in xs:
            for c in ctl_of(x):
                if c not in placed:
                    base_row(2, exps[c], 0)
                    placed.add(c)
        for x in xs:
            base_row(2, exps[x], 0)
            placed.add(x)
    for c in sorted(c for c, e in exps.items() if e["role"] == "train" and c not in placed):
        base_row(2, exps[c], 0)
        placed.add(c)

    # priority 3: FASTQ-level arms on training experiments
    for arm in FASTQ_ARMS:
        xs = [x for x in train_x if arm in p3.get(x, [])]
        for x in sorted(xs, key=lambda x: (exps[x]["assay"] not in scorable, exps[x]["assay"],
                                           exps[x]["gb"], x)):
            arm_row(3, exps[x], arm)

    # priority 4: blind experiments and their controls, then per-library and per-lane units
    blind_x = sorted((x for x, e in exps.items() if e["role"] == "blind"
                      and e["assay_class"] != CONTROL_CLASS), key=lambda x: (exps[x]["gb"], x))
    for x in blind_x:
        for c in ctl_of(x):
            if c not in placed:
                base_row(4, exps[c], 0)
                placed.add(c)
        base_row(4, exps[x], 1)
        placed.add(x)
    for x in blind_x:
        for arm in eval_arms(exps[x]):
            arm_row(4, exps[x], arm)
    tv = sorted((x for x, e in exps.items() if e["role"] in ("train", "val")
                 and e["assay_class"] != CONTROL_CLASS),
                key=lambda x: (exps[x]["role"] != "val", exps[x]["gb"], x))
    lanes = []
    for x in tv:
        e = exps[x]
        if e.get("native"):
            continue
        for b, techs in sorted(e["facts"]["reps"].items()):
            for t, r1s in sorted(techs.items()):
                lib_files = [m for f in r1s for m in mates(f, e["by_acc"])]
                if len(techs) > 1:
                    add(4, e, f"br{b}tr{t}", "base", lib_files,
                        [token(f"{x}.br{b}tr{t}.uniform.base", BASE_PRODUCTS)])
                if len(r1s) > 1:
                    for k, f in enumerate(sorted(r1s, key=lambda f: f["file"]), 1):
                        lanes.append((e, f"br{b}tr{t}ln{k}", lib_files, gb(mates(f, e["by_acc"]))))
    for e, unit, files, g in lanes:
        add(4, e, unit, "base", files, [token(f"{e['experiment']}.{unit}.uniform.base",
                                              BASE_PRODUCTS)], files_gb=g)
    for i, r in enumerate(rows, 1):
        r["row"] = str(i)

    mixed = [{"experiment": x, "eic_track": e["eic_track"], "role": e["role"], "assay": e["assay"],
              "pipeline": PIPELINE[e["assay_class"]],
              "bio_reps_mixing_se_and_pe": ",".join(map(str, e["facts"]["mixed_rep"]))}
             for x, e in sorted(exps.items()) if e["facts"]["mixed_rep"]]
    return {"rows": rows, "withheld": withheld, "excluded": excluded, "dropped": dropped_files,
            "mixed": mixed, "exps": exps, "scorable": sorted(scorable), "p3": p3,
            "train_views": {c: sorted(v) for c, v in train_views.items()}}


# ------------------------------------------------------------------------------------------ check

def views_of(rows: list[dict]) -> dict:
    """experiment -> {pooled arm: products}, from `expected_measurements`."""
    out = defaultdict(dict)
    for r in rows:
        for mid, prods in parse_tokens(r.get("expected_measurements", "")):
            exp, unit, layer, arm = mid_parts(mid)
            if unit == "pooled":
                out[exp][arm] = set(prods)
    return out


def withheld_for(withheld: list[dict], cls: str) -> set:
    out = set()
    for w in withheld:
        if w.get("assay_class", "") in ("", "*", cls):
            s = view_arm(w["source_knob"], w["source_value"])
            t = view_arm(w["target_knob"], w["target_value"])
            out.add((s, t))
    return out


def never_trained_pairs(views: dict, wh: set) -> list[tuple[str, str]]:
    """Ordered (source, target) arm pairs of one track that are withheld, not depth-only, and
    have pr1 on the source and pr2 on the target."""
    out = []
    for u, pu in views.items():
        for v, pv in views.items():
            if u != v and (u, v) in wh and not is_subset_pair(u, v) \
                    and "pr1" in pu and "pr2" in pv:
                out.append((u, v))
    return out


def check(build_rows: list[dict], ledger: list[dict], withheld: list[dict], out=print) -> int:
    S = ledger_sets(ledger)
    eic = S["eic"]
    named_controls = set(S["controls"])
    rows = build_rows
    by_exp = defaultdict(list)
    for r in rows:
        by_exp[r["experiment"]].append(r)
    views = views_of(rows)
    problems = Counter()

    out("rows per priority: " + ", ".join(f"{p}={n}" for p, n in
                                           sorted(Counter(r["priority"] for r in rows).items())))
    outside = [x for x in by_exp if x not in eic and x not in named_controls]
    out(f"experiments outside ledger: {len(outside)}" + (f"  {sorted(outside)[:10]}" if outside else ""))
    problems["outside"] = len(outside)

    role_bad = [x for x, rs in by_exp.items() if x in eic and any(r["role"] != eic[x]["role"] for r in rs)]
    out(f"EIC experiments under another role than the ledger's: {len(role_bad)}")
    problems["role"] = len(role_bad)

    match0 = {r["experiment"] for r in ledger if r.get("match") != "1"
              and r["assay_class"] != CONTROL_CLASS}
    m0 = sorted(x for x in by_exp if x in match0 and x not in eic)
    out(f"match = 0 experiments in the list: {len(m0)}")
    problems["match0"] = len(m0)

    store_only = S["store_file_experiments"] - set(eic) - named_controls
    sf = sorted(x for x in by_exp if x in store_only)
    out(f"store-file experiments that are not the published experiment: {len(sf)}")
    problems["storefile"] = len(sf)

    ctl_bad = sorted(x for x, rs in by_exp.items() if x not in eic
                     and any(r["role"] == "train" for r in rs) and "train" not in S["controls"].get(x, set()))
    out(f"training controls that control no training experiment: {len(ctl_bad)}"
        + (f"  {ctl_bad}" if ctl_bad else ""))
    problems["ctl"] = len(ctl_bad)

    blind_train = sorted(x for x, rs in by_exp.items() if x in eic and eic[x]["role"] == "blind"
                         and any(r["role"] != "blind" or r["priority"] != "4" for r in rs))
    testbed = {x for x, r in eic.items() if r.get("in_testbed") == "1"}
    out(f"blind or testbed experiments outside priority 4 or not blind: {len(blind_train)} "
        f"(testbed experiments in the list: {len(testbed & set(by_exp))}, all blind)")
    problems["blind"] = len(blind_train)

    halves_bad = sorted(x for x, r in eic.items() if r["role"] == "val" and x in views
                        and any(not {"pr1", "pr2"} <= p for p in views[x].values()))
    out(f"validation views without both pr1 and pr2: {len(halves_bad)}")
    problems["halves"] = len(halves_bad)

    wh_depth = [w for w in withheld if is_subset_pair(view_arm(w["source_knob"], w["source_value"]),
                                                       view_arm(w["target_knob"], w["target_value"]))]
    unknown = [w for w in withheld if view_arm(w["source_knob"], w["source_value"]) is None
               or view_arm(w["target_knob"], w["target_value"]) is None]
    out(f"withheld rows that are pure depth pairs (subset_pair = 1, never counted): {len(wh_depth)}")
    out(f"withheld rows naming an unknown view: {len(unknown)}")
    problems["unknown"] = len(unknown)

    short = []
    per_track = []
    for x, r in sorted(eic.items(), key=lambda kv: kv[1]["eic_track"]):
        if r["role"] != "val":
            continue
        nt = never_trained_pairs(views.get(x, {}), withheld_for(withheld, r["assay_class"]))
        per_track.append((r["eic_track"], x, r["assay_class"], len(nt)))
        if len(nt) < 2:
            short.append(r["eic_track"])
    out(f"validation tracks with < 2 never-trained non-subset pairs: {len(short)}"
        + (f"  {short}" if short else ""))
    problems["short"] = len(short)
    dist = Counter(n for *_, n in per_track)
    out("never-trained non-subset pairs per validation track: "
        + ", ".join(f"{n} pairs x {c} tracks" for n, c in sorted(dist.items())))

    # knobs on both sides of training pairs, per class
    cls_of = {r["experiment"]: r["assay_class"] for r in rows}
    role_of = {r["experiment"]: r["role"] for r in rows}
    eic_v1 = {r["experiment"] for r in rows if r["role"] == "train" and r["priority"] in ("1", "2", "3")}
    need = defaultdict(set)
    for x, v in views.items():
        if role_of.get(x) in ("val", "blind") and cls_of.get(x) != CONTROL_CLASS:
            need[cls_of[x]] |= set(v)
    for w in withheld:
        for c in ([w["assay_class"]] if w.get("assay_class", "") not in ("", "*") else list(need)):
            need[c] |= {view_arm(w["source_knob"], w["source_value"]),
                        view_arm(w["target_knob"], w["target_value"])} - {None}
    absent = []
    for c, vs in sorted(need.items()):
        wh = withheld_for(withheld, c)
        tr = [set(views[x]) for x in eic_v1 if cls_of.get(x) == c]
        for v in sorted(vs):
            src = any(v in s and any(u != v and (v, u) not in wh for u in s) for s in tr)
            tgt = any(v in s and any(u != v and (u, v) not in wh for u in s) for s in tr)
            if not (src and tgt):
                absent.append(f"{c}:{v}({'no source' if not src else ''}{'/' if not (src or tgt) else ''}"
                              f"{'no target' if not tgt else ''})")
    out(f"knobs absent from training: {len(absent)}" + (f"  {absent}" if absent else ""))
    problems["absent"] = len(absent)
    return 1 if any(problems.values()) else 0


# ------------------------------------------------------------------------------------------ probe

def probe(ledger, catalog_files, threads=16, out_path=None) -> list[dict]:
    want = wanted_experiments(ledger)
    files = [f for f in catalog_files if f["experiment"] in want
             and (f.get("file_format") == "fastq" or is_native_bam(f))]

    def head(f):
        ext = "bam" if f.get("file_format") == "bam" else "fastq.gz"
        url = f.get("url") or (f"https://www.encodeproject.org/files/{f['file']}/@@download/"
                               f"{f['file']}.{ext}")
        last = ""
        for _ in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"),
                                            timeout=30) as r:
                    return {"file": f["file"], "experiment": f["experiment"], "status": f["status"],
                            "http": str(r.status), "content_length": r.headers.get("Content-Length", ""),
                            "catalog_size": f.get("file_size", "")}
            except urllib.error.HTTPError as e:
                last = str(e.code)
                if e.code in (401, 403, 404):
                    break
            except Exception as e:  # noqa: BLE001  (network errors are recorded, not raised)
                last = type(e).__name__
        return {"file": f["file"], "experiment": f["experiment"], "status": f["status"],
                "http": last, "content_length": "", "catalog_size": f.get("file_size", "")}

    with cf.ThreadPoolExecutor(threads) as ex:
        res = sorted(ex.map(head, files), key=lambda d: d["file"])
    if out_path:
        write_tsv(out_path, PROBE_COLUMNS, res)
    return res


# ---------------------------------------------------------------------------------------- summary

STAGE_MBPS = 5.7     # measured on Nibi by D7: 32 parallel ranges per file (log 2026-11-11)


def summary(res: dict, out_dir: Path) -> None:
    rows, exps = res["rows"], res["exps"]
    out_dir.mkdir(parents=True, exist_ok=True)
    first = {}
    for r in rows:
        first.setdefault(r["experiment"], r["priority"])
    srows = []
    for p in ("1", "2", "3", "4"):
        rs = [r for r in rows if r["priority"] == p]
        xs = {r["experiment"] for r in rs}
        new = [x for x in xs if first[x] == p]
        new_gb = sum(exps[x]["gb"] for x in new)
        row_gb = sum(float(r["fastq_gb"]) for r in rs)
        h = new_gb * 1e3 / STAGE_MBPS / 3600
        srows.append({"priority": p, "rows": len(rs), "experiments": len(xs),
                      "new_experiments": len(new), "new_fastq_gb": f"{new_gb:.1f}",
                      "row_fastq_gb": f"{row_gb:.1f}",
                      "staging_h_one_stream_5.7MBps": f"{h:.1f}",
                      "staging_h_10_jobs_if_linear": f"{h / 10:.1f}",
                      "cpu_h_upper_220_per_row": f"{220 * len(rs)}",
                      "measurements": sum(len(parse_tokens(r["expected_measurements"])) for r in rs)})
        big = sorted((float(r["fastq_gb"]) for r in rs), reverse=True)
        med = sorted(float(r["fastq_gb"]) for r in rs)[len(rs) // 2] if rs else 0.0
        srows[-1]["transient_tb_50_largest_rows_x2.5"] = f"{2.5 * sum(big[:50]) / 1e3:.2f}"
        srows[-1]["transient_tb_50_median_rows_x2.5"] = f"{2.5 * 50 * med / 1e3:.2f}"
    write_tsv(out_dir / "priority_summary.tsv", list(srows[0]), srows)
    # milestones: cumulative new FASTQ GB (in row order) at which each scorable assay's validation
    # base rows (with their controls) are all built, and at which the training tracks of every
    # cell holding a scorable validation track are built
    seen, cum, ms = set(), 0.0, []
    left = defaultdict(set)
    for x, e in exps.items():
        if e["role"] == "val" and e["assay_class"] != CONTROL_CLASS and e["assay"] in res["scorable"]:
            left[e["assay"]].add(x)
    cells = {exps[x]["cell"] for xs in left.values() for x in xs}
    train_left = {x for x, e in exps.items() if e["role"] == "train"
                  and e["assay_class"] != CONTROL_CLASS and e["cell"] in cells}
    done_train = False
    for r in rows:
        x = r["experiment"]
        if x not in seen:
            seen.add(x)
            cum += exps[x]["gb"]
        if r["arm"] == "base" and r["unit"] == "pooled":
            for a in list(left):
                left[a].discard(x)
                if not left[a]:
                    ms.append({"milestone": f"validation base of scorable assay {a} complete",
                               "row": r["row"], "priority": r["priority"],
                               "cumulative_new_fastq_gb": f"{cum:.1f}",
                               "staging_h_at_5.7MBps": f"{cum * 1e3 / STAGE_MBPS / 3600:.1f}"})
                    del left[a]
            train_left.discard(x)
            if not train_left and not done_train:
                done_train = True
                ms.append({"milestone": "training tracks of every cell with a scorable validation "
                                        "track complete", "row": r["row"], "priority": r["priority"],
                           "cumulative_new_fastq_gb": f"{cum:.1f}",
                           "staging_h_at_5.7MBps": f"{cum * 1e3 / STAGE_MBPS / 3600:.1f}"})
    write_tsv(out_dir / "milestones.tsv", ("milestone", "row", "priority",
                                           "cumulative_new_fastq_gb", "staging_h_at_5.7MBps"), ms)
    cls_rows = []
    for (p, role, cls), n in sorted(Counter((r["priority"], r["role"], r["assay_class"]) for r in rows).items()):
        g = sum(float(r["fastq_gb"]) for r in rows
                if (r["priority"], r["role"], r["assay_class"]) == (p, role, cls))
        cls_rows.append({"priority": p, "role": role, "assay_class": cls, "rows": n,
                         "row_fastq_gb": f"{g:.1f}"})
    write_tsv(out_dir / "rows_by_class.tsv", list(cls_rows[0]), cls_rows)
    write_tsv(out_dir / "excluded.tsv", ("experiment", "eic_track", "role", "assay", "reason"),
              res["excluded"])
    write_tsv(out_dir / "dropped_fastqs.tsv", ("experiment", "eic_track", "role", "file", "reason"),
              res["dropped"])
    write_tsv(out_dir / "mixed_run_type.tsv", ("experiment", "eic_track", "role", "assay", "pipeline",
                                               "bio_reps_mixing_se_and_pe"), res["mixed"])
    write_tsv(out_dir / "withheld.tsv", WITHHELD_COLUMNS, res["withheld"])
    vt = []
    views = views_of(rows)
    for x, e in sorted(exps.items(), key=lambda kv: kv[1]["eic_track"]):
        if e["role"] != "val" or e["assay_class"] == CONTROL_CLASS:
            continue
        nt = never_trained_pairs(views.get(x, {}), withheld_for(res["withheld"], e["assay_class"]))
        vt.append({"eic_track": e["eic_track"], "experiment": x, "assay": e["assay"],
                   "assay_class": e["assay_class"], "paired_end": int(e["facts"]["pe"]),
                   "min_read_length": e["facts"]["min_rl"], "fastq_gb": f"{e['gb']:.2f}",
                   "arms": ",".join(sorted(views.get(x, {}))), "n_never_trained": len(nt),
                   "never_trained_pairs": ",".join(f"{u}>{v}" for u, v in nt)})
    write_tsv(out_dir / "validation_tracks.tsv", list(vt[0]), vt)
    p3 = Counter((exps[x]["assay_class"], a) for x, arms in res["p3"].items() for a in arms)
    pe = Counter((e["role"], e["assay_class"]) for e in exps.values() if e["facts"]["pe"])
    allc = Counter((e["role"], e["assay_class"]) for e in exps.values())
    cov = [{"role": r, "assay_class": c, "experiments": n, "paired_end": pe[(r, c)],
            "p3_se": p3[(c, "se")] if r == "train" else "", "p3_crop36": p3[(c, "crop36")] if r == "train" else "",
            "p3_crop50": p3[(c, "crop50")] if r == "train" else ""}
           for (r, c), n in sorted(allc.items())]
    write_tsv(out_dir / "run_type_coverage.tsv", list(cov[0]), cov)


# ------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--ledger", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=16)
    b = sub.add_parser("build")
    b.add_argument("--ledger", required=True)
    b.add_argument("--catalog", required=True)
    b.add_argument("--probe", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--withheld-out", required=True)
    b.add_argument("--summary-dir")
    c = sub.add_parser("check")
    c.add_argument("--build-list", required=True)
    c.add_argument("--ledger", required=True)
    c.add_argument("--withheld", required=True)
    a = ap.parse_args(argv)
    ledger = read_tsv(a.ledger)
    if a.cmd == "probe":
        res = probe(ledger, read_tsv(Path(a.catalog) / "files.tsv"), a.threads, a.out)
        print("FASTQs probed: %d; http: %s" % (len(res), dict(Counter(r["http"] for r in res))))
        return 0
    if a.cmd == "build":
        pr = read_probe(a.probe)
        res = build(ledger, read_tsv(Path(a.catalog) / "files.tsv"), pr)
        write_tsv(a.out, BUILD_LIST_COLUMNS, res["rows"])
        write_tsv(a.withheld_out, WITHHELD_COLUMNS, res["withheld"])
        if a.summary_dir:
            summary(res, Path(a.summary_dir))
        print(f"rows: {len(res['rows'])}; withheld rows: {len(res['withheld'])}; "
              f"excluded experiments: {len(res['excluded'])}; dropped FASTQs: {len(res['dropped'])}; "
              f"experiments mixing SE and PE in one bio rep: {len(res['mixed'])}")
        return 0
    return check(read_tsv(a.build_list), ledger, read_tsv(a.withheld))


if __name__ == "__main__":
    sys.exit(main())
