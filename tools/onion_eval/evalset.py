"""E6 -- the onion eval set (`evalset.json`) and the scorable set (PI ruling 4), from a tables dir.

    python tools/onion_eval/evalset.py build --tables <dir> [--strata <masks.h5>] [--n-regions 512]
        [--n-fit-regions 16] [--region-bins 16384] [--seed 0] [--include-blind]
        [--inputs-roles train|train,val] [--pair-splits train,val] [--regions-bed <bed>
        [--regions-exclude]] [--regions-file <json>] --out <evalset.json>
    python tools/onion_eval/evalset.py superset --base <evalset.json> --n-regions N --out <json>
    python tools/onion_eval/evalset.py derive --base <evalset.json> (--pair-regions p |
        --pair-regions-nested-in <derived evalset_p<p>.json>) --out <..._p<p>.json>
    python tools/onion_eval/evalset.py check --evalset <json> [--tables <dir>] [--allow-blind]
    python tools/onion_eval/evalset.py addon --base <frozen evalset.json> --tables <add-on tables>
        --out <json>

`build` writes `evalset.json` (pinned interface D) and, beside it, `scorable.tsv` and
`non_scorable_assays.tsv`. They go beside the evalset, not into the tables dir, because
`minicandi.dataset.tables_sha256` hashes a `scorable.tsv` found in the tables dir. The evalset
records `tables_sha256` (`minicandi.dataset.tables_sha256` of the tables at build time); the scorer
refuses tables on disk, or a run checkpoint, that do not match it (finding 9).

Blind eval set (pinned path convention). Built only with `--include-blind`, only to a file named
`evalset_blind.json` or `evalset_blind_<tag>.json` (SC): `eval/eic_v2/evalset_blind.json` from
`tables/eic_v2` (the tables that hold the blind measurements). A validation eval set is never
named so. Blind passes
over that eval set are logged in `BLIND_PASSES.tsv` beside it (score.py).

Regions (agent decision, logged). Fixed seeded random regions of `region_bins` (16384 = 2^14) bins,
each aligned to a multiple of `region_bins` (so the Haar transform of headline 1 never mixes two
regions), with a genome-mask valid fraction >= 0.9, drawn without replacement over every chromosome
except chrY and chrM. A missing `mask.h5`, or a chromosome missing from it, raises (finding 13);
the evalset records `mask_applied: true`. A second disjoint seeded draw, `fit_regions`, holds the bins on which the
scorer fits the NB size of the depth-only baseline and the floor, and the marginal NB (out of
sample: never a scored bin). ENCODE Pilot Regions are not used (eval protocol,
fixed-random-window-scope).

Imputation targets. Under the EIC split a target is the base view (layer uniform, unit pooled
preferred) of a validation track; regime `leave_one_assay_out` gives as inputs the base views of
the same cell's TRAINING tracks of every other assay (the control included); regime
`accessibility_only` (D11b.2) gives only the cell's DNase-seq or ATAC-seq training track, and a cell
without one has no target in that regime. An accessibility target has no accessibility-only item.
The ChIP-seq control (`chipseq-control`) is never an imputation target (finding 16): a validation
control goes to `control_targets` (same leave-one-assay-out inputs), which the scorer scores in a
separate table when the model predicts the control, and the control is never imputation-scorable.
Each target carries `has_halves`; a target without pseudoreplicate halves (every `store:` path,
e.g. the pilot) is kept and counted in `targets_without_halves`: the scorer gives it no noise
floor and keeps it out of headline 1 (finding 11).

Correction pairs. Every `pairs.tsv` row with rung `designed` and split val (blind only with
`--include-blind`): source = a, target = b, with `never_trained`, `subset_pair`, plus `pair_id` and
`knob` (= knobs_differing). Both sides need pseudoreplicate halves (the scorer reads the source's
pr1 and the target's pr2, PI ruling 6); a pair without them is dropped and counted.

Replicate pairs. Every pair id whose rung is a replicate-ladder rung (layers.RUNGS), any split
(the layers job reads them all; the scorer scores only the pairs of its own split).

Options (chunk SC; each one off by default, and an eval set built with the defaults is byte-identical
to one built before SC; a non-default option adds its own keys and changes no existing key):
  --inputs-roles train|train,val  `train,val` (only with --include-blind; PI ruling 21): a BLIND
                       target's inputs also hold the cell's validation tracks (its base views of
                       other assays, and its accessibility track in `accessibility_only`); a
                       validation target keeps training inputs only. Recorded as `inputs_roles`.
  --pair-splits a,b    only correction and replicate pairs whose `split` is in the list are written
                       (correction pairs also keep today's split rule); E8a passes `train,val` so a
                       blind eval set lists no blind pair. Recorded as `pair_splits`.
  --regions-bed B [--regions-exclude]  eval and fit regions are drawn only from candidate blocks
                       that lie wholly inside the (merged) BED intervals, or with --regions-exclude
                       only from blocks that overlap none (the leakage square). Recorded as
                       `regions_bed`, `regions_bed_mode`, `n_candidate_regions_before_bed`.
  A blind eval set may be named `evalset_blind.json` or `evalset_blind_<tag>.json`.
  --regions-file F     (ERx) eval and fit regions and region_bins are read from F (an eval set,
                       or a `superset` output) and used as they are; each block must be a candidate
                       block of the genome mask. Refused with --n-regions, --n-fit-regions, --seed
                       or --regions-bed; `seed` is the file's. Recorded as `regions_file`,
                       `regions_file_sha256`.

Superset (`superset`, ERx). A regions file with the base eval set's regions, its fit regions
unchanged, and the next blocks of the base's seeded candidate order that are in neither list, up
to --n-regions; so a larger eval set built from it keeps the old regions as a subset.

Derived eval set (`derive`, EVd, Amendment 8 B.4, PI ruling 42). The base eval set verbatim (the
base file is never changed), plus `pair_regions` (sorted indices into `regions`: the pair jobs run
on those regions only), `pair_regions_p`, `pair_regions_rule` and `derived_from` (path, sha256,
seed of the base). `--pair-regions p` takes indices 0, R/p, 2R/p, ... (p divides R), so p = 8, 16,
32 of 256 are nested. `--pair-regions-nested-in D` (base = a superset eval set of D's regions)
takes D's pair-region blocks plus every (R_D/p)-th of the base's other regions in their own order,
and records `pair_regions_nested_in`. The output name ends in `_p<p>.json`; a base with
`pair_regions` is refused; the blind naming rule of `build` applies.

Second experiments (OXe, Amendment 4, PI ruling 33). Add-on tables (`eic_v1x`, `eic_v2x`) hold
second experiments: rows with `in_eic = evalonly` (training and validation tracks) and the blind
track's second experiment (`in_eic = blind`), each listed in `evaluation_only.tsv` with a `use`
that starts with `other_experiment`. `build` never lists such a row as a target, an input or a
correction-pair side, under every option; its `other_experiment` pairs enter `replicate_pairs`
under the usual split rule (the layer file, `evalset_layers.json`). Tables without such rows give
a byte-identical eval set.

Add-on eval set (`addon`, decision 6). From a frozen eval set and its add-on tables: the regions,
fit regions, strata and `scorable` of the base, verbatim; `imputation_targets` = the base's
`leave_one_assay_out` targets that have an `other_experiment` pair in the add-on tables, each
followed by its second experiment as a target with the same inputs, regime and split and the key
`"evaluation_only": true`; no correction pair and no control target; `replicate_pairs` = only the
`other_experiment` pair ids whose EIC side is a chosen target (the scorer makes every listed pair
of its split a pair job); `addon_of` = sha256 of the base file, `tables_sha256` of the add-on
tables. The add-on tables must extend the base's tables (`build_info.json::addon_of_tables_sha256`
equals the base's `tables_sha256`). The scorer reads such an eval set only with `--addon`.

Read overlap (DT, Log "FOR E7a"). The testbed's arms of one track come from the same reads and
each arm's halves were split independently, so the source pr1 and the target pr2 of a correction
pair can share reads when both sides belong to one experiment and one side is a `testbed` layer
row. Such a pair gets `read_disjoint: 0` and a `read_disjoint_reason`, and the eval set the count
`correction_pairs_not_read_disjoint`; the scorer keeps it out of every correction metric. Both keys
are written only when such a pair exists.

Scorable set (ruling 4, the plan's rule), one row per assay:
  imputation_scorable  >= 3 validation tracks with pseudoreplicate halves, from >= 3 distinct cells,
                       and >= 5 training tracks
  correction_scorable  >= 2 validation tracks that each have >= 1 never-trained pair with
                       subset_pair = 0 in the evalset
  layers               per rung: the assays with >= 5 pairs from >= 3 cells
Macro averages and gate thresholds use scorable assays only; every other assay is listed in
`non_scorable_assays.tsv` (same columns) and its numbers go to a separate table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

REGION_BINS = 2 ** 14
MIN_VALID = 0.9
EXCLUDE_CHROMS = ("chrY", "chrM")
ACCESSIBILITY = ("DNase-seq", "ATAC-seq")
CONTROL = "chipseq-control"
REPLICATE_RUNGS = ("pseudoreplicate", "sequencing", "technical", "isogenic", "anisogenic",
                   "other_experiment", "other_cell_type")
IMP_MIN_VAL, IMP_MIN_CELLS, IMP_MIN_TRAIN = 3, 3, 5
COR_MIN_VAL = 2
LAYER_MIN_PAIRS, LAYER_MIN_CELLS = 5, 3
SCORABLE_COLUMNS = ("assay", "n_train", "n_val", "n_val_cells", "has_floors",
                    "imputation_scorable", "correction_scorable", "reason")
LAYOUT = "pr1 source -> pr2 target (PI ruling 6)"
BLIND_EVALSET_NAME = "evalset_blind.json"
#: a blind eval set is `evalset_blind.json` or `evalset_blind_<tag>.json` (SC)
BLIND_EVALSET_RE = re.compile(r"evalset_blind(_[A-Za-z0-9][A-Za-z0-9._-]*)?\.json")
INPUT_ROLES = (("train",), ("train", "val"))
TESTBED = "testbed"
EVALONLY = "evalonly"
#: `use` prefix of a second experiment in evaluation_only.tsv (Amendment 4, decision 2)
OTHER_EXPERIMENT_USE = "other_experiment"
NOT_READ_DISJOINT = ("one experiment's reads on both sides and a testbed arm on one side: the "
                     "testbed split each arm's halves independently, so the source pr1 and the "
                     "target pr2 can share reads (DT, Log 'FOR E7a')")


def is_blind_evalset_name(name: str) -> bool:
    return bool(BLIND_EVALSET_RE.fullmatch(str(name)))


def read_tsv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter="\t")]


def write_tsv(path, columns, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def second_experiment_ids(tables) -> set:
    """Measurement ids of the second experiments of a tables dir: `in_eic = evalonly` rows, and
    every row that `evaluation_only.tsv` marks `other_experiment` (the blind track's one has
    `in_eic = blind`). Empty for tables without them."""
    tables = Path(tables)
    ids = {r["measurement_id"] for r in read_tsv(tables / "measurements.tsv")
           if r["in_eic"] == EVALONLY}
    eo = tables / "evaluation_only.tsv"
    if eo.is_file():
        ids |= {r["measurement_id"] for r in read_tsv(eo)
                if (r.get("use") or "").strip().startswith(OTHER_EXPERIMENT_USE)}
    return ids


def _is_base(r: dict) -> bool:
    return (r.get("arm_knob") or "").strip() in ("", "base")


def _base_rank(r: dict) -> tuple:
    return ({"uniform": 0, "native": 1, "testbed": 2}.get(r.get("layer", ""), 9),
            0 if r.get("unit") == "pooled" else 1, r["measurement_id"])


def _halves(r: dict) -> bool:
    """True when the row has pseudoreplicate halves on disk (a `store:` path never has them)."""
    return (str(r.get("has_pseudoreps", "")).strip() in ("1", "true", "True")
            and not str(r.get("counts_path", "")).startswith("store:"))


def base_views(rows: list[dict], roles) -> dict:
    """(cell, assay) -> the preferred base view row among rows whose in_eic is in `roles`."""
    out: dict = {}
    for r in rows:
        if r["in_eic"] in roles and _is_base(r):
            k = (r["cell"], r["assay"])
            if k not in out or _base_rank(r) < _base_rank(out[k]):
                out[k] = r
    return out


# --------------------------------------------------------------------------------------- regions

def genome_dir(tables_dir: Path) -> Path:
    return Path(tables_dir).parent.parent / "genome"


def read_chrom_sizes(path: Path) -> dict:
    """chrom -> length in bp from a chrom_sizes.json in either layout: flat `{"chr1": 248956422}`
    or nested `{"assembly": "GRCh38", "chrom_sizes": {"chr1": 248956422}}` (the cluster store)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    sizes = raw.get("chrom_sizes", raw) if isinstance(raw, dict) else None
    if not isinstance(sizes, dict):
        raise ValueError(f"{path}: not a chrom -> size map")
    return {str(c): int(n) for c, n in sizes.items()}


def candidate_blocks(gdir: Path, region_bins: int, min_valid: float = MIN_VALID) -> list:
    """Every aligned block [chrom, start_bin] of `region_bins` bins with valid fraction >= min_valid."""
    import h5py
    sizes = read_chrom_sizes(gdir / "chrom_sizes.json")
    mpath = gdir / "mask.h5"
    out = []
    if not mpath.is_file():
        raise FileNotFoundError(f"{mpath}: no genome mask; regions would be drawn unmasked")
    with h5py.File(mpath, "r") as mf:
        for chrom in sorted(sizes):
            if chrom in EXCLUDE_CHROMS:
                continue
            if chrom not in mf:
                raise KeyError(f"{mpath}: no mask for {chrom}; regions would be drawn unmasked")
            nb = sizes[chrom] // 25
            m = np.asarray(mf[chrom][:], bool)
            for s in range(0, nb - region_bins + 1, region_bins):
                if m[s:s + region_bins].mean() >= min_valid:
                    out.append([chrom, int(s)])
    return out


def read_bed(path) -> dict:
    """chrom -> merged, sorted [(start_bp, end_bp)] of a BED file (0-based, half-open); header,
    track and comment lines are skipped; overlapping or touching intervals are merged."""
    raw = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        f = line.split("\t") if "\t" in line else line.split()
        if not f or f[0].startswith(("#", "track", "browser")) or len(f) < 3:
            continue
        s, e = int(f[1]), int(f[2])
        if e < s:
            raise ValueError(f"{path}: interval end before start: {line!r}")
        raw[f[0]].append((s, e))
    out = {}
    for c, iv in raw.items():
        merged = []
        for s, e in sorted(iv):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out[c] = [(s, e) for s, e in merged]
    return out


def filter_blocks(cands: list, bed: dict, region_bins: int, exclude: bool = False) -> list:
    """Candidate blocks wholly inside one merged BED interval, or (exclude) overlapping none."""
    import bisect
    out = []
    for c, sb in cands:
        s, e = int(sb) * 25, (int(sb) + region_bins) * 25
        iv = bed.get(c, [])
        i = bisect.bisect_right([a for a, _ in iv], s) - 1     # last interval starting <= s
        inside = i >= 0 and iv[i][1] >= e
        j = bisect.bisect_left([a for a, _ in iv], e)          # intervals starting before e
        overlaps = any(b > s for _, b in iv[max(0, i):j]) if j > 0 else False
        if (not exclude and inside) or (exclude and not overlaps):
            out.append([c, sb])
    return out


def draw_regions(cands: list, n: int, n_fit: int, seed: int) -> tuple[list, list]:
    """Seeded draw without replacement: n eval regions, then n_fit disjoint fit regions; each list
    sorted by (chrom, start) so the scorer's concatenation order is fixed."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cands))
    ev = [cands[i] for i in order[:n]]
    fit = [cands[i] for i in order[n:n + n_fit]]
    return sorted(ev), sorted(fit)


# --------------------------------------------------------------------------------------- items

def input_pool(rows: list[dict], inputs_roles, target_role: str) -> dict:
    """(cell, assay) -> the base view a target of role `target_role` may take as input: training
    tracks, plus (inputs_roles = train,val, blind targets only; ruling 21) validation tracks of a
    (cell, assay) with no training track."""
    pool = base_views(rows, ("train",))
    if "val" in inputs_roles and target_role == "blind":
        for k, r in base_views(rows, ("val",)).items():
            pool.setdefault(k, r)
    return pool


def imputation_targets(rows: list[dict], splits, inputs_roles=("train",)) -> list[dict]:
    pools = {role: input_pool(rows, inputs_roles, role) for role in ("val", "blind")}
    targets = base_views(rows, splits)
    items = []
    for (cell, assay), r in sorted(targets.items()):
        if r["in_eic"] == "train" or assay == CONTROL:
            continue
        train = pools[r["in_eic"]]
        loao = sorted(t["measurement_id"] for (c, a), t in train.items() if c == cell and a != assay)
        items.append({"target": r["measurement_id"], "inputs": loao,
                      "regime": "leave_one_assay_out", "split": r["in_eic"],
                      "assay": assay, "cell": cell, "has_halves": _halves(r)})
        if assay in ACCESSIBILITY:
            continue
        acc = [train[(cell, a)]["measurement_id"] for a in ACCESSIBILITY if (cell, a) in train]
        if acc:
            items.append({"target": r["measurement_id"], "inputs": acc[:1],
                          "regime": "accessibility_only", "split": r["in_eic"],
                          "assay": assay, "cell": cell, "has_halves": _halves(r)})
    return items


def control_targets(rows: list[dict], splits, inputs_roles=("train",)) -> list[dict]:
    """Validation (or blind) ChIP-seq control tracks, each with the cell's training tracks of the
    other assays as inputs; scored apart from the imputation headline (finding 16)."""
    pools = {role: input_pool(rows, inputs_roles, role) for role in ("val", "blind")}
    out = []
    for (cell, assay), r in sorted(base_views(rows, splits).items()):
        if r["in_eic"] == "train" or assay != CONTROL:
            continue
        train = pools[r["in_eic"]]
        loao = sorted(t["measurement_id"] for (c, a), t in train.items() if c == cell and a != assay)
        out.append({"target": r["measurement_id"], "inputs": loao, "regime": "control",
                    "split": r["in_eic"], "assay": assay, "cell": cell, "has_halves": _halves(r)})
    return out


def shares_reads(a: dict, b: dict) -> bool:
    """True when the pr1 half of `a` and the pr2 half of `b` can share reads: one experiment on
    both sides and a testbed-layer row on one side (the testbed split each arm's halves on its own)."""
    same = (a.get("experiment") or "") != "" and a.get("experiment") == b.get("experiment")
    return same and TESTBED in (a.get("layer"), b.get("layer"))


def correction_pairs(pairs: list[dict], meas: dict, splits, pair_splits=None) -> tuple[list, int]:
    out, dropped = [], 0
    for p in pairs:
        if p["rung"] != "designed" or p["split"] not in splits:
            continue
        if pair_splits is not None and p["split"] not in pair_splits:
            continue
        a, b = meas.get(p["a"]), meas.get(p["b"])
        if a is None or b is None or not (_halves(a) and _halves(b)):
            dropped += 1
            continue
        item = {"source": p["a"], "target": p["b"], "never_trained": int(p["never_trained"] or 0),
                "subset_pair": int(p["subset_pair"] or 0), "split": p["split"],
                "pair_id": p["pair_id"], "knob": p.get("knobs_differing", ""),
                "assay": p["assay"], "cell": p["cell"]}
        if shares_reads(a, b):
            item.update(read_disjoint=0, read_disjoint_reason=NOT_READ_DISJOINT)
        out.append(item)
    return out, dropped


# --------------------------------------------------------------------------------------- scorable

def scorable_rows(rows: list[dict], cor: list[dict], splits=("val",)) -> list[dict]:
    train = base_views(rows, ("train",))
    val = base_views(rows, splits)
    val = {k: r for k, r in val.items() if r["in_eic"] != "train"}
    assays = sorted({r["assay"] for r in rows})
    nt_tracks = defaultdict(set)
    for p in cor:
        if p["never_trained"] == 1 and p["subset_pair"] == 0 and p.get("read_disjoint", 1):
            nt_tracks[p["assay"]].add(p["cell"])
    out = []
    for a in assays:
        n_train = sum(1 for (c, aa) in train if aa == a)
        v = [r for (c, aa), r in val.items() if aa == a]
        with_h = [r for r in v if _halves(r)]
        cells = {r["cell"] for r in with_h}
        imp = len(with_h) >= IMP_MIN_VAL and len(cells) >= IMP_MIN_CELLS and n_train >= IMP_MIN_TRAIN
        corr = len(nt_tracks[a] & {r["cell"] for r in v}) >= COR_MIN_VAL
        why = []
        if a == CONTROL:
            imp = False
            why.append("the ChIP-seq control is not an imputation target (scored apart when the "
                       "model predicts it)")
        if len(with_h) < IMP_MIN_VAL:
            why.append(f"{len(with_h)} validation tracks with halves (need {IMP_MIN_VAL})")
        if len(cells) < IMP_MIN_CELLS:
            why.append(f"{len(cells)} validation cells (need {IMP_MIN_CELLS})")
        if n_train < IMP_MIN_TRAIN:
            why.append(f"{n_train} training tracks (need {IMP_MIN_TRAIN})")
        if not corr:
            why.append(f"{len(nt_tracks[a])} validation tracks with a never-trained non-subset pair "
                       f"(need {COR_MIN_VAL})")
        out.append({"assay": a, "n_train": n_train, "n_val": len(v), "n_val_cells": len(cells),
                    "has_floors": int(bool(v) and len(with_h) == len(v)),
                    "imputation_scorable": int(imp), "correction_scorable": int(corr),
                    "reason": "; ".join(why) if why else "scorable"})
    return out


def layer_scorable(pairs: list[dict], ids: list) -> dict:
    keep = set(ids)
    cnt, cells = defaultdict(int), defaultdict(set)
    for p in pairs:
        if p["pair_id"] in keep:
            cnt[(p["rung"], p["assay"])] += 1
            cells[(p["rung"], p["assay"])].add(p["cell"])
    out = {r: [] for r in REPLICATE_RUNGS}
    for (r, a), n in sorted(cnt.items()):
        if n >= LAYER_MIN_PAIRS and len(cells[(r, a)]) >= LAYER_MIN_CELLS:
            out[r].append(a)
    return out


# --------------------------------------------------------------------------------------- build

def read_regions_file(path) -> dict:
    """The `regions`, `fit_regions`, `region_bins` (and `seed`, when present) of a regions file:
    an eval set, or the output of `superset` (ERx)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    miss = [k for k in ("regions", "fit_regions", "region_bins") if k not in raw]
    if miss:
        raise ValueError(f"{path}: regions file lacks {miss}")
    blocks = lambda xs: [[str(c), int(s)] for c, s in xs]  # noqa: E731
    return {"regions": blocks(raw["regions"]), "fit_regions": blocks(raw["fit_regions"]),
            "region_bins": int(raw["region_bins"]), "seed": raw.get("seed")}


def file_regions(rf: dict, cands: list, path) -> tuple[list, list]:
    """The file's eval and fit regions as they are; each must be a candidate block, none twice."""
    ok = set(map(tuple, cands))
    seen: set = set()
    for name in ("regions", "fit_regions"):
        for b in rf[name]:
            if tuple(b) not in ok:
                raise ValueError(f"{path}: {name} block {b[0]}:{b[1]} is not a candidate block of "
                                 f"the genome mask at {rf['region_bins']} bins")
            if tuple(b) in seen:
                raise ValueError(f"{path}: block {b[0]}:{b[1]} is listed twice")
            seen.add(tuple(b))
    return rf["regions"], rf["fit_regions"]


def superset(base, n_regions: int, out) -> dict:
    """A regions file that holds the base eval set's regions, its fit regions unchanged, and the
    next blocks of the base's seeded candidate order that are in neither list, up to n_regions
    (ERx). `regions` is sorted as `draw_regions` sorts; `added_regions` keeps permutation order."""
    base, out = Path(base), Path(out)
    raw = base.read_bytes()
    bev = json.loads(raw.decode("utf-8"))
    if bev.get("regions_bed"):
        raise ValueError(f"{base}: built with --regions-bed; superset supports unfiltered "
                         f"candidate blocks only")
    if bev.get("seed") is None:
        raise ValueError(f"{base}: no seed; the candidate order cannot be reproduced")
    rb, seed = int(bev["region_bins"]), int(bev["seed"])
    cands = candidate_blocks(genome_dir(tables_dir_of(bev, base)), rb)
    if bev.get("n_candidate_regions") is not None and len(cands) != int(bev["n_candidate_regions"]):
        raise ValueError(f"{base}: {bev['n_candidate_regions']} candidate blocks at build, "
                         f"{len(cands)} now (the genome mask changed)")
    rf = read_regions_file(base)
    regions, fit = file_regions(rf, cands, base)
    if n_regions < len(regions):
        raise ValueError(f"n_regions {n_regions} < the base's {len(regions)} regions")
    taken = set(map(tuple, regions)) | set(map(tuple, fit))
    added = []
    for i in np.random.default_rng(seed).permutation(len(cands)):
        if len(regions) + len(added) >= n_regions:
            break
        if tuple(cands[i]) not in taken:
            added.append(cands[i])
    if len(regions) + len(added) < n_regions:
        raise ValueError(f"{len(cands)} candidate blocks: cannot reach {n_regions} regions "
                         f"beside {len(fit)} fit regions")
    res = {"regions": sorted(regions + added), "fit_regions": fit, "region_bins": rb, "seed": seed,
           "n_candidate_regions": len(cands), "added_regions": added,
           "superset_of": str(base.resolve()), "superset_of_sha256": hashlib.sha256(raw).hexdigest()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def build(tables, out, *, strata=None, n_regions=None, n_fit_regions=None, region_bins=None,
          seed=None, include_blind=False, inputs_roles=("train",), pair_splits=None,
          regions_bed=None, regions_exclude=False, regions_file=None) -> dict:
    """`n_regions`, `n_fit_regions`, `seed` default to 512, 16, 0 and `region_bins` to 2^14 without
    `regions_file`; with it (ERx) they come from the file and the draw options are refused."""
    from minicandi.dataset import tables_sha256
    rf = None
    if regions_file is not None:
        given = [n for n, v in (("n_regions", n_regions), ("n_fit_regions", n_fit_regions),
                                ("seed", seed), ("regions_bed", regions_bed)) if v is not None]
        if regions_exclude:
            given.append("regions_exclude")
        if given:
            raise ValueError(f"--regions-file is refused together with {given}")
        rf = read_regions_file(regions_file)
        if region_bins is not None and int(region_bins) != rf["region_bins"]:
            raise ValueError(f"region_bins {region_bins} differs from the regions file's "
                             f"{rf['region_bins']}")
        region_bins, seed = rf["region_bins"], rf["seed"]
    else:
        n_regions = 512 if n_regions is None else n_regions
        n_fit_regions = 16 if n_fit_regions is None else n_fit_regions
        region_bins = REGION_BINS if region_bins is None else region_bins
        seed = 0 if seed is None else seed
    tables = Path(tables).resolve()
    out = Path(out)
    if include_blind != is_blind_evalset_name(out.name):
        raise ValueError(f"a blind eval set is built with --include-blind into a file named "
                         f"{BLIND_EVALSET_NAME} or evalset_blind_<tag>.json, and only then (got "
                         f"include_blind={include_blind}, {out.name})")
    if region_bins <= 0 or region_bins & (region_bins - 1):
        raise ValueError("region_bins must be a power of 2")
    inputs_roles = tuple(inputs_roles)
    if inputs_roles not in INPUT_ROLES:
        raise ValueError(f"inputs_roles must be one of {[','.join(r) for r in INPUT_ROLES]}")
    if "val" in inputs_roles and not include_blind:
        raise ValueError("--inputs-roles train,val is allowed only with --include-blind (ruling 21: "
                         "a blind target's inputs may hold the cell's validation tracks)")
    if pair_splits is not None:
        pair_splits = tuple(pair_splits)
        bad = [s for s in pair_splits if s not in ("train", "val", "blind")]
        if bad or not pair_splits:
            raise ValueError(f"pair_splits: unknown split(s) {bad} (train, val, blind)")
    if regions_exclude and not regions_bed:
        raise ValueError("--regions-exclude needs --regions-bed")
    rows = read_tsv(tables / "measurements.tsv")
    pairs = read_tsv(tables / "pairs.tsv")
    second = second_experiment_ids(tables)
    if second:               # OXe: never a target, an input or a correction-pair side
        rows = [r for r in rows if r["measurement_id"] not in second]
        cor_pairs = [p for p in pairs if p["a"] not in second and p["b"] not in second]
    else:
        cor_pairs = pairs
    meas = {r["measurement_id"]: r for r in rows}
    splits = ("val", "blind") if include_blind else ("val",)

    cands = candidate_blocks(genome_dir(tables), region_bins)
    n_before_bed = len(cands)
    if regions_bed:
        cands = filter_blocks(cands, read_bed(regions_bed), region_bins, exclude=regions_exclude)
        if len(cands) < n_regions + n_fit_regions:
            raise ValueError(f"{regions_bed}: {len(cands)} candidate blocks "
                             f"{'outside' if regions_exclude else 'inside'} the BED intervals, "
                             f"need {n_regions} + {n_fit_regions}")
    if rf is not None:
        regions, fit_regions = file_regions(rf, cands, regions_file)
    else:
        regions, fit_regions = draw_regions(cands, n_regions, n_fit_regions, seed)
    imp = imputation_targets(rows, splits, inputs_roles)
    ctl = control_targets(rows, splits, inputs_roles)
    cor, dropped = correction_pairs(cor_pairs, meas, splits, pair_splits)
    rep = [p["pair_id"] for p in pairs if p["rung"] in REPLICATE_RUNGS and p["a"] != p["b"]
           and (pair_splits is None or p["split"] in pair_splits)]
    sc = scorable_rows(rows, cor)
    names = []
    if strata and Path(strata).is_file():
        import h5py
        with h5py.File(strata, "r") as f:
            names = sorted(f.keys())
    ev = {
        "tables": tables.name, "tables_dir": str(tables), "tables_sha256": tables_sha256(tables),
        "include_blind": bool(include_blind), "mask_applied": True,
        "regions": regions, "fit_regions": fit_regions, "region_bins": int(region_bins),
        "seed": None if seed is None else int(seed), "n_candidate_regions": len(cands),
        "imputation_targets": imp, "control_targets": ctl,
        "targets_without_halves": sorted({t["target"] for t in imp if not t["has_halves"]}),
        "correction_pairs": cor, "correction_pairs_dropped": dropped,
        "correction_layout": LAYOUT, "replicate_pairs": rep,
        "scorable": {"imputation": [r["assay"] for r in sc if r["imputation_scorable"]],
                     "correction": [r["assay"] for r in sc if r["correction_scorable"]],
                     "layers": layer_scorable(pairs, rep)},
        "strata": names, "strata_path": str(Path(strata).resolve()) if names else "",
    }
    # SC options: keys only when an option is not at its default (default build byte-identical)
    if inputs_roles != ("train",):
        ev["inputs_roles"] = list(inputs_roles)
    if pair_splits is not None:
        ev["pair_splits"] = list(pair_splits)
    if regions_bed:
        ev.update(regions_bed=str(Path(regions_bed).resolve()),
                  regions_bed_mode="exclude" if regions_exclude else "inside",
                  n_candidate_regions_before_bed=n_before_bed)
    if rf is not None:       # ERx: key only with --regions-file (default build byte-identical)
        ev.update(regions_file=str(Path(regions_file).resolve()),
                  regions_file_sha256=sha256_file(regions_file))
    n_shared = sum(1 for p in cor if p.get("read_disjoint", 1) == 0)
    if n_shared:
        ev["correction_pairs_not_read_disjoint"] = n_shared
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    write_tsv(out.parent / "scorable.tsv", SCORABLE_COLUMNS, sc)
    write_tsv(out.parent / "non_scorable_assays.tsv", SCORABLE_COLUMNS,
              [r for r in sc if not (r["imputation_scorable"] and r["correction_scorable"])])
    return ev


def tables_dir_of(ev: dict, evalset_path) -> Path:
    if ev.get("tables_dir") and Path(ev["tables_dir"]).is_dir():
        return Path(ev["tables_dir"])
    return Path(evalset_path).resolve().parent.parent.parent / "tables" / ev["tables"]


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(evalset, tables=None, allow_blind=False) -> list[str]:
    """Problems of an evalset (empty list = passes); prints the counts per scorable assay."""
    from minicandi.dataset import tables_sha256
    ev = json.loads(Path(evalset).read_text(encoding="utf-8"))
    tdir = Path(tables) if tables else tables_dir_of(ev, evalset)
    meas = {r["measurement_id"]: r for r in read_tsv(tdir / "measurements.tsv")}
    bad = []
    if not ev.get("tables_sha256"):
        bad.append("evalset has no tables_sha256 (rebuild it)")
    elif ev["tables_sha256"] != tables_sha256(tdir):
        bad.append(f"tables_sha256 of {tdir} differs from the evalset's")
    if not ev.get("mask_applied"):
        bad.append("evalset regions were not drawn under the genome mask")
    for t in ev["imputation_targets"]:
        if meas.get(t["target"], {}).get("assay") == CONTROL:
            bad.append(f"control {t['target']} is an imputation target")
    for t in ev["imputation_targets"]:
        if not t["inputs"]:
            bad.append(f"target {t['target']} ({t['regime']}) has no input")
        for m in [t["target"]] + t["inputs"]:
            if m not in meas:
                bad.append(f"{m} not in measurements.tsv")
        val_ok = ("val" in ev.get("inputs_roles", ["train"])
                  and meas.get(t["target"], {}).get("in_eic") == "blind")
        for m in t["inputs"]:
            if m in meas and meas[m]["in_eic"] != "train" and not (
                    val_ok and meas[m]["in_eic"] == "val"):
                bad.append(f"input {m} of {t['target']} is not a training track")
    second = second_experiment_ids(tdir)
    flagged = [t for t in ev["imputation_targets"] if t.get("evaluation_only")]
    for t in ev["imputation_targets"]:
        if t["target"] in second and not t.get("evaluation_only"):
            bad.append(f"second experiment {t['target']} is a target without evaluation_only")
        for m in t["inputs"]:
            if m in second:
                bad.append(f"second experiment {m} is an input of {t['target']}")
    for t in flagged:
        if not ev.get("addon_of"):
            bad.append(f"evaluation-only target {t['target']} in an eval set without addon_of")
        if t["target"] not in second:
            bad.append(f"evaluation-only target {t['target']} is not a second experiment")
    for t in ev.get("control_targets", []):
        if t["target"] in second or set(t["inputs"]) & second:
            bad.append(f"second experiment in control target {t['target']}")
    for p in ev["correction_pairs"]:
        if {p["source"], p["target"]} & second:
            bad.append(f"second experiment in correction pair {p.get('pair_id', '')}")
    if ev.get("addon_of"):
        bad += addon_problems(ev, tdir)
    ids = [t["target"] for t in ev["imputation_targets"]] + \
          [m for p in ev["correction_pairs"] for m in (p["source"], p["target"])]
    if not allow_blind:
        bad += [f"blind row {m}" for m in sorted(set(ids)) if m in meas and meas[m]["in_eic"] == "blind"]
    rb = int(ev.get("region_bins", REGION_BINS))
    for c, s in ev["regions"]:
        if int(s) % rb:
            bad.append(f"region {c}:{s} is not aligned to {rb} bins")
    if set(map(tuple, ev["regions"])) & set(map(tuple, ev.get("fit_regions", []))):
        bad.append("fit regions overlap eval regions")
    if "pair_regions" in ev:                  # EVd: a derived eval set
        bad += pair_regions_problems(ev)
    n_imp = defaultdict(int)
    for t in ev["imputation_targets"]:
        if t["regime"] == "leave_one_assay_out":
            n_imp[t["assay"]] += 1
    n_cor = defaultdict(int)
    for p in ev["correction_pairs"]:
        if p["never_trained"] == 1 and p["subset_pair"] == 0:
            n_cor[p["assay"]] += 1
    print(f"regions: {len(ev['regions'])} x {rb} bins; fit regions: {len(ev.get('fit_regions', []))}")
    for a in ev["scorable"]["imputation"]:
        print(f"imputation-scorable {a}: {n_imp[a]} validation targets")
    for a in ev["scorable"]["correction"]:
        print(f"correction-scorable {a}: {n_cor[a]} never-trained non-subset pairs")
    print(f"problems: {len(bad)}")
    for b in bad[:50]:
        print(f"  {b}")
    return bad


# --------------------------------------------------------------------------------------- derive

PAIR_REGIONS_NAME_RE = re.compile(r".*_p([1-9][0-9]*)\.json")
PAIR_REGIONS_RULE = ("pair_regions = the indices 0, R/p, 2R/p, ... of the region order (`regions` "
                     "as stored); nested for p = 8, 16, 32 of R = 256 (PI ruling 42, Amendment 8 B.4)")
PAIR_REGIONS_NESTED_RULE = ("pair_regions = the nested base's pair-region blocks, plus every "
                            "(R_base/p)-th of this eval set's other regions in their own region "
                            "order, starting at the first (Amendment 8 B.4)")


def pair_positions(n_regions: int, p: int) -> list[int]:
    """Indices 0, R/p, 2R/p, ... of a region order of length R; p must divide R."""
    if p <= 0 or n_regions % p:
        raise ValueError(f"--pair-regions {p} must be positive and divide the {n_regions} regions")
    step = n_regions // p
    return list(range(0, n_regions, step))


def pair_regions_problems(ev: dict) -> list[str]:
    """Problems of the `pair_regions` keys of a derived eval set (empty list = passes)."""
    bad, pr, n = [], ev["pair_regions"], len(ev["regions"])
    if not isinstance(pr, list) or not pr or not all(isinstance(i, int) for i in pr):
        return ["pair_regions is not a non-empty list of region indices"]
    if pr != sorted(set(pr)):
        bad.append("pair_regions is not strictly increasing")
    if pr[0] < 0 or pr[-1] >= n:
        bad.append(f"pair_regions index out of range of the {n} regions")
    df = ev.get("derived_from")
    if not isinstance(df, dict) or not df.get("path") or not df.get("sha256"):
        bad.append("pair_regions without derived_from (path, sha256)")
    elif Path(df["path"]).is_file() and sha256_file(df["path"]) != df["sha256"]:
        bad.append(f"derived_from {df['path']}: sha256 differs (the base changed)")
    return bad


def derive(base, out, *, pair_regions=None, nested_in=None) -> dict:
    """The base eval set plus `pair_regions` and `derived_from` (EVd); every other key verbatim.
    `pair_regions=p`: indices 0, R/p, ... of the region order. `nested_in=<derived eval set>`: the
    base is a superset of that eval set's regions; its pair regions are that eval set's pair-region
    blocks plus every (R_nested/p)-th of the base's other regions in their own order."""
    base, out = Path(base), Path(out)
    if (pair_regions is None) == (nested_in is None):
        raise ValueError("give exactly one of --pair-regions and --pair-regions-nested-in")
    if out.resolve() == base.resolve():
        raise ValueError(f"{out}: the base eval set is never overwritten")
    raw = base.read_bytes()
    bev = json.loads(raw.decode("utf-8"))
    if "pair_regions" in bev:
        raise ValueError(f"{base}: already has pair_regions (derive from the underived eval set)")
    if bool(bev.get("include_blind")) != is_blind_evalset_name(out.name):
        raise ValueError(f"a blind eval set derives only into a file named "
                         f"{BLIND_EVALSET_NAME[:-5]}_<tag>.json, and only then (got "
                         f"include_blind={bool(bev.get('include_blind'))}, {out.name})")
    regions = [tuple(b) for b in bev["regions"]]
    nested = None
    if nested_in is None:
        p = int(pair_regions)
        idx = pair_positions(len(regions), p)
        rule = PAIR_REGIONS_RULE
    else:
        nraw = Path(nested_in).read_bytes()
        nev = json.loads(nraw.decode("utf-8"))
        if "pair_regions" not in nev or "pair_regions_p" not in nev:
            raise ValueError(f"{nested_in}: not a derived eval set (no pair_regions)")
        if int(nev["region_bins"]) != int(bev["region_bins"]) or \
                nev["fit_regions"] != bev["fit_regions"]:
            raise ValueError(f"{nested_in}: region_bins or fit_regions differ from {base}'s")
        pos = {b: i for i, b in enumerate(regions)}
        nreg = [tuple(b) for b in nev["regions"]]
        missing = [b for b in nreg if b not in pos]
        if missing:
            raise ValueError(f"{base}: not a superset of {nested_in} ({len(missing)} regions "
                             f"missing, e.g. {missing[0][0]}:{missing[0][1]})")
        p = int(nev["pair_regions_p"])
        step = len(nreg) // p
        old = set(nreg)
        others = [i for i, b in enumerate(regions) if b not in old]
        idx = sorted({pos[nreg[i]] for i in nev["pair_regions"]} | set(others[::step]))
        rule = PAIR_REGIONS_NESTED_RULE
        nested = {"path": str(Path(nested_in).resolve()),
                  "sha256": hashlib.sha256(nraw).hexdigest()}
    m = PAIR_REGIONS_NAME_RE.fullmatch(out.name)
    if not m or int(m.group(1)) != p:
        raise ValueError(f"{out.name}: a derived eval set's name ends in _p{p}.json")
    ev = dict(bev)
    ev.update(pair_regions=idx, pair_regions_p=p, pair_regions_rule=rule,
              derived_from={"path": str(base.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                            "seed": bev.get("seed")})
    if nested is not None:
        ev["pair_regions_nested_in"] = nested
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    return ev


# --------------------------------------------------------------------------------------- add-on

def addon_info(tables) -> dict:
    """`build_info.json` of add-on tables ({} when absent)."""
    p = Path(tables) / "build_info.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def addon_problems(ev: dict, tdir) -> list[str]:
    """Problems specific to an add-on eval set (empty list = passes)."""
    bad = []
    if ev.get("correction_pairs"):
        bad.append("an add-on eval set lists correction pairs")
    if ev.get("control_targets"):
        bad.append("an add-on eval set lists control targets")
    pairs = {p["pair_id"]: p for p in read_tsv(Path(tdir) / "pairs.tsv")}
    meas = {r["measurement_id"]: r for r in read_tsv(Path(tdir) / "measurements.tsv")}
    eic = {(meas[t["target"]]["cell"], meas[t["target"]]["assay"]) for t in ev["imputation_targets"]
           if not t.get("evaluation_only") and t["target"] in meas}
    for pid in ev.get("replicate_pairs", []):
        p = pairs.get(pid)
        if (p is None or p["rung"] != "other_experiment"
                or not {(meas.get(m, {}).get("cell"), meas.get(m, {}).get("assay"))
                        for m in (p["a"], p["b"])} & eic):
            bad.append(f"replicate pair {pid} is not an other_experiment pair of a listed target")
    info = addon_info(tdir)
    if not info.get("addon_of_tables_sha256"):
        bad.append(f"{tdir}/build_info.json has no addon_of_tables_sha256 (not add-on tables)")
    return bad


def addon(base, tables, out) -> dict:
    """The add-on eval set of a frozen eval set on its add-on tables (module docstring)."""
    from minicandi.dataset import tables_sha256
    base, tables, out = Path(base), Path(tables).resolve(), Path(out)
    raw = base.read_bytes()
    bev = json.loads(raw.decode("utf-8"))
    if bev.get("addon_of"):
        raise ValueError(f"{base}: already an add-on eval set")
    info = addon_info(tables)
    if info.get("addon_of_tables_sha256") != bev.get("tables_sha256"):
        raise ValueError(f"{tables}/build_info.json: addon_of_tables_sha256 "
                         f"{str(info.get('addon_of_tables_sha256'))[:12]} is not the base eval set's "
                         f"tables_sha256 {str(bev.get('tables_sha256'))[:12]}")
    meas = {r["measurement_id"]: r for r in read_tsv(tables / "measurements.tsv")}
    second = second_experiment_ids(tables)
    other = defaultdict(list)          # EIC track (cell, assay) -> [(pair id, second experiment)]
    for p in read_tsv(tables / "pairs.tsv"):
        if p["rung"] != "other_experiment" or p["a"] == p["b"]:
            continue
        for e, x in ((p["a"], p["b"]), (p["b"], p["a"])):
            if x in second and e not in second and e in meas:
                other[(meas[e]["cell"], meas[e]["assay"])].append((p["pair_id"], x))
    targets, pair_ids, seen = [], [], set()
    for t in bev["imputation_targets"]:
        row = meas.get(t["target"])
        if row is None:
            raise ValueError(f"base target {t['target']} is not in {tables}/measurements.tsv")
        key = (row["cell"], row["assay"])
        if t["regime"] != "leave_one_assay_out" or key not in other:
            continue
        if set(t["inputs"]) & second:
            raise ValueError(f"base target {t['target']} has a second experiment as input")
        targets.append(dict(t))
        for pid, x in sorted(other[key]):
            if pid not in seen:
                seen.add(pid)
                pair_ids.append(pid)
            targets.append(dict(t, target=x, has_halves=_halves(meas[x]), evaluation_only=True))
    ev = dict(bev)
    for k in ("correction_pairs_not_read_disjoint", "pair_splits"):
        ev.pop(k, None)
    ev.update({
        "tables": tables.name, "tables_dir": str(tables), "tables_sha256": tables_sha256(tables),
        "imputation_targets": targets, "control_targets": [],
        "targets_without_halves": sorted({t["target"] for t in targets
                                          if not t["has_halves"] and not t.get("evaluation_only")}),
        "correction_pairs": [], "correction_pairs_dropped": 0, "replicate_pairs": pair_ids,
        "addon_of": hashlib.sha256(raw).hexdigest(), "addon_of_path": str(base.resolve()),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    return ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tables", required=True)
    b.add_argument("--strata")
    b.add_argument("--n-regions", type=int, default=None, help="default 512")
    b.add_argument("--n-fit-regions", type=int, default=None, help="default 16")
    b.add_argument("--region-bins", type=int, default=None, help=f"default {REGION_BINS}")
    b.add_argument("--seed", type=int, default=None, help="default 0")
    b.add_argument("--regions-file", default=None,
                   help="json with regions, fit_regions, region_bins, used as they are (ERx)")
    b.add_argument("--include-blind", action="store_true")
    b.add_argument("--inputs-roles", default="train", choices=("train", "train,val"))
    b.add_argument("--pair-splits", default=None,
                   help="comma list of splits; only correction and replicate pairs of these splits")
    b.add_argument("--regions-bed", default=None)
    b.add_argument("--regions-exclude", action="store_true")
    b.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--evalset", required=True)
    c.add_argument("--tables")
    c.add_argument("--allow-blind", action="store_true")
    x = sub.add_parser("addon")
    x.add_argument("--base", required=True)
    x.add_argument("--tables", required=True)
    x.add_argument("--out", required=True)
    u = sub.add_parser("superset")
    u.add_argument("--base", required=True)
    u.add_argument("--n-regions", type=int, required=True)
    u.add_argument("--out", required=True)
    d = sub.add_parser("derive")
    d.add_argument("--base", required=True)
    g = d.add_mutually_exclusive_group(required=True)
    g.add_argument("--pair-regions", type=int, default=None)
    g.add_argument("--pair-regions-nested-in", default=None)
    d.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "derive":
        ev = derive(a.base, a.out, pair_regions=a.pair_regions, nested_in=a.pair_regions_nested_in)
        print(f"derived evalset: {len(ev['pair_regions'])} of {len(ev['regions'])} regions are pair "
              f"regions (p = {ev['pair_regions_p']}) -> {a.out}")
        return 0
    if a.cmd == "superset":
        res = superset(a.base, a.n_regions, a.out)
        print(f"regions file: {len(res['regions'])} regions ({len(res['added_regions'])} added), "
              f"{len(res['fit_regions'])} fit regions -> {a.out}")
        return 0
    if a.cmd == "addon":
        ev = addon(a.base, a.tables, a.out)
        n_x = sum(1 for t in ev["imputation_targets"] if t.get("evaluation_only"))
        print(f"add-on evalset: {len(ev['imputation_targets']) - n_x} validation targets with a "
              f"second experiment, {n_x} evaluation-only targets, {len(ev['replicate_pairs'])} "
              f"other_experiment pair ids -> {a.out}")
        return 0
    if a.cmd == "build":
        ev = build(a.tables, a.out, strata=a.strata, n_regions=a.n_regions,
                   n_fit_regions=a.n_fit_regions, region_bins=a.region_bins, seed=a.seed,
                   include_blind=a.include_blind, inputs_roles=tuple(a.inputs_roles.split(",")),
                   pair_splits=(tuple(s for s in a.pair_splits.split(",") if s)
                                if a.pair_splits is not None else None),
                   regions_bed=a.regions_bed, regions_exclude=a.regions_exclude,
                   regions_file=a.regions_file)
        print(f"evalset: {len(ev['regions'])} regions, {len(ev['imputation_targets'])} imputation "
              f"items ({len(ev['targets_without_halves'])} targets without halves), "
              f"{len(ev['control_targets'])} control targets, "
              f"{len(ev['correction_pairs'])} correction pairs "
              f"({ev['correction_pairs_dropped']} dropped: no halves), "
              f"{len(ev['replicate_pairs'])} replicate pairs; imputation-scorable "
              f"{len(ev['scorable']['imputation'])}, correction-scorable "
              f"{len(ev['scorable']['correction'])} -> {a.out}")
        return 0
    return 1 if check(a.evalset, a.tables, a.allow_blind) else 0


if __name__ == "__main__":
    sys.exit(main())
