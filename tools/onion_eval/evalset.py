"""E6 -- the onion eval set (`evalset.json`) and the scorable set (PI ruling 4), from a tables dir.

    python tools/onion_eval/evalset.py build --tables <dir> [--strata <masks.h5>] [--n-regions 512]
        [--n-fit-regions 16] [--region-bins 16384] [--seed 0] [--include-blind] --out <evalset.json>
    python tools/onion_eval/evalset.py check --evalset <json> [--tables <dir>] [--allow-blind]

`build` writes `evalset.json` (pinned interface D) and, beside it, `scorable.tsv` and
`non_scorable_assays.tsv`. They go beside the evalset, not into the tables dir, because
`minicandi.dataset.tables_sha256` hashes a `scorable.tsv` found in the tables dir.

Regions (agent decision, logged). Fixed seeded random regions of `region_bins` (16384 = 2^14) bins,
each aligned to a multiple of `region_bins` (so the Haar transform of headline 1 never mixes two
regions), with a genome-mask valid fraction >= 0.9, drawn without replacement over every chromosome
except chrY and chrM. A second disjoint seeded draw, `fit_regions`, holds the bins on which the
scorer fits the NB size of the depth-only baseline and the floor, and the marginal NB (out of
sample: never a scored bin). ENCODE Pilot Regions are not used (eval protocol,
fixed-random-window-scope).

Imputation targets. Under the EIC split a target is the base view (layer uniform, unit pooled
preferred) of a validation track; regime `leave_one_assay_out` gives as inputs the base views of
the same cell's TRAINING tracks of every other assay (the control included); regime
`accessibility_only` (D11b.2) gives only the cell's DNase-seq or ATAC-seq training track, and a cell
without one has no target in that regime. An accessibility target has no accessibility-only item.

Correction pairs. Every `pairs.tsv` row with rung `designed` and split val (blind only with
`--include-blind`): source = a, target = b, with `never_trained`, `subset_pair`, plus `pair_id` and
`knob` (= knobs_differing). Both sides need pseudoreplicate halves (the scorer reads the source's
pr1 and the target's pr2, PI ruling 6); a pair without them is dropped and counted.

Replicate pairs. Every pair id whose rung is a replicate-ladder rung (layers.RUNGS), any split
(the layers job reads them all; the scorer scores only the pairs of its own split).

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


def read_tsv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter="\t")]


def write_tsv(path, columns, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def _is_base(r: dict) -> bool:
    return (r.get("arm_knob") or "").strip() in ("", "base")


def _base_rank(r: dict) -> tuple:
    return ({"uniform": 0, "native": 1, "testbed": 2}.get(r.get("layer", ""), 9),
            0 if r.get("unit") == "pooled" else 1, r["measurement_id"])


def _halves(r: dict) -> bool:
    return str(r.get("has_pseudoreps", "")).strip() in ("1", "true", "True")


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


def candidate_blocks(gdir: Path, region_bins: int, min_valid: float = MIN_VALID) -> list:
    """Every aligned block [chrom, start_bin] of `region_bins` bins with valid fraction >= min_valid."""
    import h5py
    sizes = json.loads((gdir / "chrom_sizes.json").read_text(encoding="utf-8"))
    mpath = gdir / "mask.h5"
    out = []
    mf = h5py.File(mpath, "r") if mpath.is_file() else None
    try:
        for chrom in sorted(sizes):
            if chrom in EXCLUDE_CHROMS:
                continue
            nb = int(sizes[chrom]) // 25
            m = np.asarray(mf[chrom][:], bool) if (mf is not None and chrom in mf) else None
            for s in range(0, nb - region_bins + 1, region_bins):
                if m is None or m[s:s + region_bins].mean() >= min_valid:
                    out.append([chrom, int(s)])
    finally:
        if mf is not None:
            mf.close()
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

def imputation_targets(rows: list[dict], splits) -> list[dict]:
    train = base_views(rows, ("train",))
    targets = base_views(rows, splits)
    items = []
    for (cell, assay), r in sorted(targets.items()):
        if r["in_eic"] == "train":
            continue
        loao = sorted(t["measurement_id"] for (c, a), t in train.items() if c == cell and a != assay)
        items.append({"target": r["measurement_id"], "inputs": loao,
                      "regime": "leave_one_assay_out", "split": r["in_eic"],
                      "assay": assay, "cell": cell})
        if assay in ACCESSIBILITY:
            continue
        acc = [train[(cell, a)]["measurement_id"] for a in ACCESSIBILITY if (cell, a) in train]
        if acc:
            items.append({"target": r["measurement_id"], "inputs": acc[:1],
                          "regime": "accessibility_only", "split": r["in_eic"],
                          "assay": assay, "cell": cell})
    return items


def correction_pairs(pairs: list[dict], meas: dict, splits) -> tuple[list, int]:
    out, dropped = [], 0
    for p in pairs:
        if p["rung"] != "designed" or p["split"] not in splits:
            continue
        a, b = meas.get(p["a"]), meas.get(p["b"])
        if a is None or b is None or not (_halves(a) and _halves(b)):
            dropped += 1
            continue
        out.append({"source": p["a"], "target": p["b"], "never_trained": int(p["never_trained"] or 0),
                    "subset_pair": int(p["subset_pair"] or 0), "split": p["split"],
                    "pair_id": p["pair_id"], "knob": p.get("knobs_differing", ""),
                    "assay": p["assay"], "cell": p["cell"]})
    return out, dropped


# --------------------------------------------------------------------------------------- scorable

def scorable_rows(rows: list[dict], cor: list[dict], splits=("val",)) -> list[dict]:
    train = base_views(rows, ("train",))
    val = base_views(rows, splits)
    val = {k: r for k, r in val.items() if r["in_eic"] != "train"}
    assays = sorted({r["assay"] for r in rows})
    nt_tracks = defaultdict(set)
    for p in cor:
        if p["never_trained"] == 1 and p["subset_pair"] == 0:
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

def build(tables, out, *, strata=None, n_regions=512, n_fit_regions=16, region_bins=REGION_BINS,
          seed=0, include_blind=False) -> dict:
    tables = Path(tables).resolve()
    out = Path(out)
    if region_bins <= 0 or region_bins & (region_bins - 1):
        raise ValueError("region_bins must be a power of 2")
    rows = read_tsv(tables / "measurements.tsv")
    meas = {r["measurement_id"]: r for r in rows}
    pairs = read_tsv(tables / "pairs.tsv")
    splits = ("val", "blind") if include_blind else ("val",)

    cands = candidate_blocks(genome_dir(tables), region_bins)
    regions, fit_regions = draw_regions(cands, n_regions, n_fit_regions, seed)
    imp = imputation_targets(rows, splits)
    cor, dropped = correction_pairs(pairs, meas, splits)
    rep = [p["pair_id"] for p in pairs if p["rung"] in REPLICATE_RUNGS and p["a"] != p["b"]]
    sc = scorable_rows(rows, cor)
    names = []
    if strata and Path(strata).is_file():
        import h5py
        with h5py.File(strata, "r") as f:
            names = sorted(f.keys())
    ev = {
        "tables": tables.name, "tables_dir": str(tables),
        "regions": regions, "fit_regions": fit_regions, "region_bins": int(region_bins),
        "seed": int(seed), "n_candidate_regions": len(cands),
        "imputation_targets": imp, "correction_pairs": cor, "correction_pairs_dropped": dropped,
        "correction_layout": LAYOUT, "replicate_pairs": rep,
        "scorable": {"imputation": [r["assay"] for r in sc if r["imputation_scorable"]],
                     "correction": [r["assay"] for r in sc if r["correction_scorable"]],
                     "layers": layer_scorable(pairs, rep)},
        "strata": names, "strata_path": str(Path(strata).resolve()) if names else "",
    }
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
    ev = json.loads(Path(evalset).read_text(encoding="utf-8"))
    tdir = Path(tables) if tables else tables_dir_of(ev, evalset)
    meas = {r["measurement_id"]: r for r in read_tsv(tdir / "measurements.tsv")}
    bad = []
    for t in ev["imputation_targets"]:
        if not t["inputs"]:
            bad.append(f"target {t['target']} ({t['regime']}) has no input")
        for m in [t["target"]] + t["inputs"]:
            if m not in meas:
                bad.append(f"{m} not in measurements.tsv")
        for m in t["inputs"]:
            if m in meas and meas[m]["in_eic"] != "train":
                bad.append(f"input {m} of {t['target']} is not a training track")
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tables", required=True)
    b.add_argument("--strata")
    b.add_argument("--n-regions", type=int, default=512)
    b.add_argument("--n-fit-regions", type=int, default=16)
    b.add_argument("--region-bins", type=int, default=REGION_BINS)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--include-blind", action="store_true")
    b.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--evalset", required=True)
    c.add_argument("--tables")
    c.add_argument("--allow-blind", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        ev = build(a.tables, a.out, strata=a.strata, n_regions=a.n_regions,
                   n_fit_regions=a.n_fit_regions, region_bins=a.region_bins, seed=a.seed,
                   include_blind=a.include_blind)
        print(f"evalset: {len(ev['regions'])} regions, {len(ev['imputation_targets'])} imputation "
              f"items, {len(ev['correction_pairs'])} correction pairs "
              f"({ev['correction_pairs_dropped']} dropped: no halves), "
              f"{len(ev['replicate_pairs'])} replicate pairs; imputation-scorable "
              f"{len(ev['scorable']['imputation'])}, correction-scorable "
              f"{len(ev['scorable']['correction'])} -> {a.out}")
        return 0
    return 1 if check(a.evalset, a.tables, a.allow_blind) else 0


if __name__ == "__main__":
    sys.exit(main())
