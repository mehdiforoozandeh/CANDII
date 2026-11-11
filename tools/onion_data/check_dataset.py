"""t152 chunk D6t -- the check that guards a tables directory built by `build_tables.py`.

    python check_dataset.py --tables <dir> [--superset-of <older dir>] [--ledger L]
        [--published P] [--genome-dir G] [--workers N] [--width-regions 4]
        [--width-max-pairs 30] [--no-widths]

It prints one count per rule and exits 1 when any count is nonzero (2 on a missing table file):
  measurements outside ledger                  experiment not an EIC ledger experiment with
                                               match = 1, nor a control one of them names
  accessions outside the published EIC list    a non-control row whose experiment, track and role
                                               are not a row of the challenge's published list
                                               (`tests/fixtures/eic_published_accessions.tsv`); a
                                               control row not named as a control by a published
                                               experiment (the published list holds no controls)
  rows whose EIC split differs from the ledger `in_eic` != the ledger role
  validation or blind measurement in a training pair
  pairs whose split is not their sides' EIC role   (the more restricted of the two)
  withheld combinations in training pairs      a `withheld.tsv` combination on a split-train pair
  never_trained flags that disagree with withheld.tsv
  blind rows readable in train or val mode     a row that must be blind-only (ledger role blind,
                                               listed in `evaluation_only.tsv`, or layer testbed)
                                               whose `in_eic` is a role `OnionDataset` reads in
                                               mode train or val (`dataset.ROLES`)
  testbed rows not flagged evaluation-only     a layer-testbed row missing from evaluation_only.tsv
  pairs between two testbed arms in pairs.tsv
  arms whose halves do not derive from the parent split   a side of a designed pair whose halves
                                               were not cut from the same read-level split as the
                                               other side (`build_tables.SplitSignature`)
  sha256 mismatches                            counts.h5 missing, or its sha256 != the row's
  pairs naming an unknown measurement, duplicate ids
  rows of the older tables changed or missing  (with --superset-of)
Then one line per replicate rung (`layers.RUNGS`) and assay: the pair count over every split and
the width of the 95% interval of the rung's mean pair variance (depth-matched log1p difference,
`layers.pair_variance`, every Haar band), cluster bootstrap over cells (`layers._interval`), from at
most `--width-max-pairs` seeded training and validation pairs on `--width-regions` seeded masked
regions of 16384 bins. It is a coverage check of the tables, not the onion layers report (chunk
E8a). Blind pairs are counted, never read. A rung with no pair prints
`0 pairs, not populated by EIC`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from minicandi import contract as C  # noqa: E402
from minicandi.dataset import (MEASUREMENT_COLUMNS, PAIR_COLUMNS, ROLES,  # noqa: E402
                               WITHHELD_COLUMNS, Withheld, read_tsv)
from onion_data import build_list as BL  # noqa: E402
from onion_data import build_tables as BT  # noqa: E402

PUBLISHED = TOOLS.parent / "tests" / "fixtures" / "eic_published_accessions.tsv"
LEDGER = "/project/def-maxwl/mforooz/ONION/eic/accession_ledger.tsv"
REPLICATE_RUNGS = ("pseudoreplicate", "sequencing", "technical", "isogenic", "anisogenic",
                   "other_experiment", "other_cell_type")
N_BOOT = 200


def sha256_path(path: str):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def published_index(path) -> dict:
    from onion_data import eic_accessions as EA
    pub = EA.load_published(path)
    return {(r["experiment"], r["eic_track"], r["role"]) for r in pub}, {r["experiment"] for r in pub}


# --------------------------------------------------------------------------------------------
# interval widths


def pick_regions(genome_dir: Path, rows: list, n: int, L: int, seed: int) -> list:
    """`n` seeded aligned blocks of L bins: masked candidates from the genome layer when it has a
    mask, else every aligned block of the first readable counts.h5."""
    cands = []
    if (genome_dir / "mask.h5").is_file() and (genome_dir / "chrom_sizes.json").is_file():
        from onion_eval.evalset import candidate_blocks
        cands = candidate_blocks(genome_dir, L)
    else:
        import h5py
        for r in rows:
            d = BT.mdir_of(r)
            if d is not None and (d / "counts.h5").is_file():
                with h5py.File(d / "counts.h5", "r") as f:
                    for c in sorted(f.keys()):
                        if c not in ("chrY", "chrM"):
                            cands += [[c, s] for s in range(0, f[c].shape[0] - L + 1, L)]
                break
    if not cands:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cands))[:n]
    return sorted([cands[i][0], int(cands[i][1])] for i in idx)


def _read(path: Path, regions, L):
    import h5py
    with h5py.File(path, "r") as f:
        out = []
        for c, s in regions:
            v = np.zeros(L, np.int64)
            if c in f:
                x = np.asarray(f[c][s:s + L])
                v[:x.size] = x
            out.append(v)
        depth = float(f.attrs.get("depth_reads", 0) or 0)
    return np.stack(out), depth


def rung_widths(by_id: dict, pairs: list, regions: list, max_pairs: int, seed: int) -> dict:
    """(rung, assay) -> {"width", "n_read", "n_cells_read"} on non-blind pairs."""
    from onion_eval import layers as LY
    L = 2 ** LY.J
    groups = defaultdict(list)
    for p in pairs:
        if p["rung"] in REPLICATE_RUNGS and p["split"] != "blind":
            groups[(p["rung"], p["assay"])].append(p)
    out = {}
    for key in sorted(groups):
        ps = sorted(groups[key], key=lambda p: p["pair_id"])
        rng = np.random.default_rng([seed, zlib.crc32("|".join(key).encode())])
        ps = [ps[i] for i in sorted(rng.permutation(len(ps))[:max_pairs])]
        recs = []
        for p in ps:
            a, b = by_id.get(p["a"]), by_id.get(p["b"])
            da, db = (BT.mdir_of(a) if a else None), (BT.mdir_of(b) if b else None)
            if da is None or db is None:
                continue
            try:
                if p["a"] == p["b"]:
                    x, dx = _read(da / "pr1.h5", regions, L)
                    y, dy = _read(da / "pr2.h5", regions, L)
                else:
                    x, _ = _read(da / "counts.h5", regions, L)
                    y, _ = _read(db / "counts.h5", regions, L)
                    dx, dy = float(a["depth_reads"] or 0), float(b["depth_reads"] or 0)
                if dx <= 0 or dy <= 0:
                    continue
                pv = LY.pair_variance(x, y, dx, dy, seed=zlib.crc32(p["pair_id"].encode()))
            except (OSError, KeyError, ValueError):
                continue
            recs.append((p["cell"] or a["cell"], pv["variance"]["all"]["all"]))
        if not recs:
            out[key] = {"width": float("nan"), "n_read": 0, "n_cells_read": 0}
            continue
        cells = sorted({c for c, _ in recs})
        v = np.array([x for _, x in recs])
        ci = np.array([cells.index(c) for c, _ in recs])
        est = float(v.mean())
        boot = []
        brng = np.random.default_rng([seed, 1, zlib.crc32("|".join(key).encode())])
        for _ in range(N_BOOT):
            pick = np.bincount(brng.integers(0, len(cells), len(cells)), minlength=len(cells))
            w = pick[ci].astype(float)
            boot.append(float((w * v).sum() / w.sum()) if w.sum() else np.nan)
        lo, hi = LY._interval(est, boot, len(cells))
        out[key] = {"width": hi - lo, "n_read": len(recs), "n_cells_read": len(cells),
                    "variance": est}
    return out


# --------------------------------------------------------------------------------------------
# check


def check(a) -> int:
    T = Path(a.tables)
    need = (BT.MEASUREMENTS, BT.PAIRS, BT.WITHHELD, BT.COVARIATES)
    lack = [n for n in need if not (T / n).is_file()]
    if lack:
        print(f"FAIL missing table files: {lack}")
        return 2
    info = json.loads((T / BT.BUILD_INFO).read_text()) if (T / BT.BUILD_INFO).is_file() else {}
    rows = read_tsv(T / BT.MEASUREMENTS, MEASUREMENT_COLUMNS)
    pairs = read_tsv(T / BT.PAIRS, PAIR_COLUMNS)
    shared = read_tsv(T / BT.SHARED, PAIR_COLUMNS) if (T / BT.SHARED).is_file() else []
    withheld = Withheld(read_tsv(T / BT.WITHHELD, WITHHELD_COLUMNS))
    eval_only = ({r["measurement_id"]: r["use"] for r in
                  read_tsv(T / BT.EVAL_ONLY, BT.EVAL_ONLY_COLUMNS)}
                 if (T / BT.EVAL_ONLY).is_file() else {})
    ledger_path = a.ledger or info.get("ledger") or LEDGER
    L = BT.ledger_index(BL.read_tsv(ledger_path))
    pub_keys, pub_exps = published_index(a.published)

    by_id = {}
    dup = 0
    for r in rows:
        dup += r["measurement_id"] in by_id
        by_id[r["measurement_id"]] = r
    V = {}                                           # rule -> count (all must be 0)
    ex = defaultdict(list)                           # rule -> examples

    def bad(rule, example=None):
        V[rule] = V.get(rule, 0) + 1
        if example is not None and len(ex[rule]) < 3:
            ex[rule].append(example)

    rules = ("measurements outside ledger", "accessions outside the published EIC list",
             "rows whose EIC split differs from the ledger",
             "validation or blind measurement in a training pair",
             "pairs whose split is not their sides' EIC role",
             "withheld combinations in training pairs",
             "never_trained flags that disagree with withheld.tsv",
             "blind rows readable in train or val mode",
             "testbed rows not flagged evaluation-only",
             "pairs between two testbed arms in pairs.tsv",
             "arms whose halves do not derive from the parent split",
             "sha256 mismatches", "pairs naming an unknown measurement", "duplicate ids")
    for k in rules:
        V[k] = 0
    V["duplicate ids"] = dup

    # controls a published experiment names (the published list itself holds none)
    pub_controls = set()
    for x, r in L["eic"].items():
        if x in pub_exps:
            pub_controls |= set(BL.split_acc(r.get("encode_possible_controls", "")))
    readable = set(ROLES["train"]) | set(ROLES["val"])
    for r in rows:
        mid, x = r["measurement_id"], r["experiment"]
        role = L["role"].get(x)
        if role is None:
            bad("measurements outside ledger", mid)
        elif r["in_eic"] != role:
            bad("rows whose EIC split differs from the ledger", f"{mid} {r['in_eic']} != {role}")
        if r["assay"] == C.CONTROL:
            if x not in pub_controls:
                bad("accessions outside the published EIC list", mid)
        elif (x, r["eic_track"], r["in_eic"]) not in pub_keys:
            bad("accessions outside the published EIC list", f"{mid} ({r['eic_track']}, {r['in_eic']})")
        blind_only = role == "blind" or mid in eval_only or r["layer"] == "testbed"
        if blind_only and r["in_eic"] in readable:
            bad("blind rows readable in train or val mode", f"{mid} in_eic={r['in_eic']}")
        if r["layer"] == "testbed" and mid not in eval_only:
            bad("testbed rows not flagged evaluation-only", mid)
    for mid in eval_only:
        if mid not in by_id:
            bad("pairs naming an unknown measurement", f"evaluation_only.tsv: {mid}")

    sig = BT.SplitSignature()
    not_derived, seen_ids = set(), set()
    for p in pairs:
        if p["pair_id"] in seen_ids:
            bad("duplicate ids", p["pair_id"])
        seen_ids.add(p["pair_id"])
        A, B = by_id.get(p["a"]), by_id.get(p["b"])
        if A is None or B is None:
            bad("pairs naming an unknown measurement", p["pair_id"])
            continue
        split = BT.pair_split(A, B)
        if p["split"] != split:
            bad("pairs whose split is not their sides' EIC role", f"{p['pair_id']} {p['split']} != {split}")
        if p["split"] == "train" and (A["in_eic"] != "train" or B["in_eic"] != "train"):
            bad("validation or blind measurement in a training pair", p["pair_id"])
        blocked = p["rung"] == "designed" and withheld.blocks(A, B)
        if blocked and p["split"] == "train":
            bad("withheld combinations in training pairs", p["pair_id"])
        want_nt = "1" if blocked and p["split"] in ("val", "blind") else "0"
        if p["never_trained"] != want_nt:
            bad("never_trained flags that disagree with withheld.tsv", p["pair_id"])
        if p["rung"] == "designed":
            if A["layer"] == "testbed" and B["layer"] == "testbed":
                bad("pairs between two testbed arms in pairs.tsv", p["pair_id"])
            elif not sig.one_split(A, B):
                not_derived |= {p["a"], p["b"]}
    V["arms whose halves do not derive from the parent split"] = len(not_derived)
    ex["arms whose halves do not derive from the parent split"] = sorted(not_derived)[:3]

    paths = sorted({r["counts_path"][3:] for r in rows if r["counts_path"].startswith("h5:")})
    if a.workers > 1 and len(paths) > 1:
        with Pool(a.workers) as pool:
            got = dict(zip(paths, pool.map(sha256_path, paths, chunksize=4)))
    else:
        got = {p: sha256_path(p) for p in paths}
    n_store = 0
    for r in rows:
        if not r["counts_path"].startswith("h5:"):
            n_store += 1
            continue
        h = got[r["counts_path"][3:]]
        if h is None or h != r["sha256"]:
            bad("sha256 mismatches", f"{r['measurement_id']}" + (" (no file)" if h is None else ""))

    if a.superset_of:
        old = Path(a.superset_of)
        n_bad = 0
        for name in (BT.MEASUREMENTS, BT.PAIRS):
            new = set(BT.body_lines(T / name))
            n_bad += sum(1 for ln in BT.body_lines(old / name) if ln not in new)
        if (old / BT.WITHHELD).read_bytes() != (T / BT.WITHHELD).read_bytes():
            n_bad += 1
        ov = json.loads((old / BT.COVARIATES).read_text()).get("vocab", {})
        nv = json.loads((T / BT.COVARIATES).read_text()).get("vocab", {})
        n_bad += sum(1 for f, vals in ov.items() if nv.get(f, [])[:len(vals)] != list(vals))
        V["rows of the older tables changed or missing"] = n_bad

    # ---- report
    print(f"tables: {T}  ledger: {ledger_path}  published: {a.published}")
    sp = Counter(r["in_eic"] for r in rows)
    print(f"measurements: {len(rows)} (" + ", ".join(f"{s} {sp.get(s, 0)}" for s in BT.ROLE_ORDER)
          + f"); evaluation-only rows: {len(eval_only)}; store: rows (no sha256): {n_store}")
    print(f"withheld combinations: {len(withheld)} rows; never-trained pairs: "
          + ", ".join(f"{s} {sum(1 for p in pairs if p['never_trained'] == '1' and p['split'] == s)}"
                      for s in ("val", "blind")))
    print(f"pairs: {len(pairs)}; pairs_shared_reads.tsv: {len(shared)}")
    for k, n in V.items():
        print(f"{k}: {n}" + (f"  e.g. {', '.join(ex[k])}" if n and ex.get(k) else ""))

    pc = defaultdict(list)
    for p in pairs:
        if p["rung"] in REPLICATE_RUNGS:
            pc[(p["rung"], p["assay"])].append(p)
    widths, regions = {}, []
    if not a.no_widths:
        from onion_eval import layers as LY
        gdir = Path(a.genome_dir) if a.genome_dir else T.parent.parent / "genome"
        regions = pick_regions(gdir, rows, a.width_regions, 2 ** LY.J, a.seed)
        if regions:
            widths = rung_widths(by_id, pairs, regions, a.width_max_pairs, a.seed)
    assays = [t for t in C.TRACKS if any(r["assay"] == t for r in rows)]
    print(f"replicate rungs ({len(regions)} regions of 16384 bins for the widths"
          + ("; widths off" if a.no_widths or not regions else "") + "):")
    for rung in REPLICATE_RUNGS:
        for assay in assays:
            ps = pc.get((rung, assay), [])
            if not ps:
                print(f"rung {rung} {assay}: 0 pairs, not populated by EIC")
                continue
            cells = {p["cell"] or by_id.get(p["a"], {}).get("cell", "") for p in ps}
            nb = sum(1 for p in ps if p["split"] == "blind")
            w = widths.get((rung, assay))
            ws = ("interval width n/a" if w is None or w["n_read"] == 0 else
                  f"interval width {w['width']:.4g} (mean pair variance {w['variance']:.4g}; "
                  f"{w['n_read']} pairs from {w['n_cells_read']} cells read)")
            print(f"rung {rung} {assay}: {len(ps)} pairs, {len(cells)} cells"
                  + (f" ({nb} blind, not read)" if nb else "") + f", {ws}")
    n_viol = sum(V.values())
    print(f"check: {'PASS' if not n_viol else f'FAIL ({n_viol} violations)'}")
    return 0 if not n_viol else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", required=True)
    ap.add_argument("--superset-of", default="")
    ap.add_argument("--ledger", default="", help="default: build_info.json's ledger, else Nibi's")
    ap.add_argument("--published", default=str(PUBLISHED))
    ap.add_argument("--genome-dir", default="", help="default: <tables>/../../genome")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--width-regions", type=int, default=4)
    ap.add_argument("--width-max-pairs", type=int, default=30)
    ap.add_argument("--no-widths", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return check(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
