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
Add-on tables (Amendment 4, chunk OXt; `--addon-of` and `--other-experiments` default to
`build_info.json`). A second-experiment row is a row with `in_eic = evalonly` or whose experiment
is in the reviewed list `$OXL`; such an experiment is exempt from the two ledger lines above only
under decision 3, which these lines enforce (each must be 0, and each prints on every tables):
  second-experiment rows outside the reviewed list
  second-experiment rows not flagged evaluation-only     `in_eic` not `evalonly` (`blind` only for
                                               a blind track)
  second-experiment rows that are not a pooled uniform base view   `<x>.pooled.uniform.base`, or
                                               `<x>.pooled.native.base` for `$OXL` source
                                               `redacted_bam` (`build_tables.second_view_layer`)
  second-experiment rows in a training pair or outside the other_experiment rung   (pairs.tsv and
                                               pairs_shared_reads.tsv; both sides second: also)
  evaluation-only rows readable in a training mode   mode train must raise RoleError in the
                                               constructor (3 configs); in mode val, the
                                               second-experiment ids in `train_tracks`,
                                               `train_base`, or as the `_base_view` of another row
  evalonly rows that disagree with evaluation_only.tsv   either side missing, or another `use`
  rows of the base tables changed or missing   a base file not byte-identical, or (for the four
                                               appended files) not a byte prefix
and the count line `second-experiment rows: N (other lab n1, same lab n2)`.
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
from minicandi.dataset import (EVALONLY, MEASUREMENT_COLUMNS, PAIR_COLUMNS,  # noqa: E402
                               ROLES, WITHHELD_COLUMNS, Withheld, read_tsv)
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
# add-on: second experiments (Amendment 4)

ADDON_RULES = ("second-experiment rows outside the reviewed list",
               "second-experiment rows not flagged evaluation-only",
               "second-experiment rows that are not a pooled uniform base view",
               "second-experiment rows in a training pair or outside the other_experiment rung",
               "evaluation-only rows readable in a training mode",
               "evalonly rows that disagree with evaluation_only.tsv",
               "rows of the base tables changed or missing")
TRAIN_CONFIGS = ({}, {"final_model": True}, {"train_on_validation": True, "final_model": True})


def load_reviewed_list(a, info: dict) -> dict:
    path = a.other_experiments or info.get("other_experiments", "")
    return BT.reviewed_list(BL.read_tsv(path)) if path else {}


def second_row(r: dict, oxl: dict) -> bool:
    return r["in_eic"] == EVALONLY or r["experiment"] in oxl


def _dataset_genome(T: Path, a, tmp: Path) -> str:
    """A genome dir for the guard probe: the real one when it has chrom sizes, else a stand-in
    with one chromosome (the probe reads no counts and no sequence)."""
    g = Path(a.genome_dir) if a.genome_dir else T.parent.parent / "genome"
    if (g / "chrom_sizes.json").is_file():
        return str(g)
    (tmp / "chrom_sizes.json").write_text(json.dumps({"chr1": 4096 * C.RES}))
    return str(tmp)


def dataset_probe(T: Path, a, sx_ids: set, bad) -> None:
    """`evaluation-only rows readable in a training mode`, through `OnionDataset` itself."""
    import tempfile
    from minicandi import dataset as D
    rule = "evaluation-only rows readable in a training mode"
    with tempfile.TemporaryDirectory() as td:
        base_cfg = {"dna": "off", "genome_dir": _dataset_genome(T, a, Path(td)),
                    "context_bins": 64}
        for cfg in TRAIN_CONFIGS:
            try:
                D.OnionDataset(T, "train", dict(base_cfg, **cfg))
            except D.RoleError:
                continue
            except Exception as e:  # noqa: BLE001 - any other outcome is not the refusal
                bad(rule, f"mode train {cfg}: {type(e).__name__} instead of RoleError")
                continue
            bad(rule, f"mode train {cfg}: constructed")
        try:
            ds = D.OnionDataset(T, "val", base_cfg)
        except Exception as e:  # noqa: BLE001
            bad(rule, f"mode val: {type(e).__name__}: {e}")
            return
        for k, mids in sorted(ds.train_tracks.items()):
            for m in sorted(set(mids) & sx_ids):
                bad(rule, f"{m} in train_tracks{k}")
        for k, m in sorted(ds.train_base.items()):
            if m in sx_ids:
                bad(rule, f"{m} is train_base{k}")
        for mid, r in sorted(ds.rows.items()):
            if mid not in sx_ids and ds._base_view(r) in sx_ids:
                bad(rule, f"{ds._base_view(r)} is the base view of {mid}")


def base_changed(T: Path, base: Path) -> int:
    n = 0
    for p in sorted(base.iterdir()):
        if not p.is_file() or p.name == BT.BUILD_INFO:
            continue
        q = T / p.name
        old = p.read_bytes()
        new = q.read_bytes() if q.is_file() else None
        if p.name in BT.ADDON_APPENDED:
            if new is None or not new.startswith(old):
                have = set(BT.body_lines(q)) if new is not None else set()
                n += max(1, sum(1 for ln in BT.body_lines(p) if ln not in have))
        elif new != old:
            n += 1
    return n


def addon_rules(a, T: Path, info: dict, rows: list, pairs: list, eval_only: dict, L: dict,
                bad, V: dict) -> str:
    """The add-on lines (module doc); returns the count line."""
    for k in ADDON_RULES:
        V[k] = 0
    oxl = load_reviewed_list(a, info)
    sx = [r for r in rows if second_row(r, oxl)]
    sx_ids = {r["measurement_id"] for r in sx}
    n_other = n_same = 0
    for r in sx:
        mid, x = r["measurement_id"], r["experiment"]
        e = oxl.get(x)
        if e is None or L["role"].get(x) is not None:
            bad("second-experiment rows outside the reviewed list", mid)
        else:
            n_other += "lab" in e["pair_class"].split(";")
            n_same += "lab" not in e["pair_class"].split(";")
        blind_ok = r["in_eic"] == "blind" and e is not None and e["split"] == "blind"
        if r["in_eic"] != EVALONLY and not blind_ok:
            bad("second-experiment rows not flagged evaluation-only", f"{mid} in_eic={r['in_eic']}")
        layer = BT.second_view_layer(e["source"]) if e is not None else "uniform"
        if not BT.is_second_view(r, layer):
            bad("second-experiment rows that are not a pooled uniform base view",
                f"{mid} (allowed: {x}.pooled.{layer}.base)")
        if eval_only.get(mid) != BT.USE_OTHER_EXPERIMENT:
            bad("evalonly rows that disagree with evaluation_only.tsv",
                f"{mid}: " + ("not listed" if mid not in eval_only else "use differs"))
    by_id = {r["measurement_id"]: r for r in rows}
    for mid, use in eval_only.items():
        if use == BT.USE_OTHER_EXPERIMENT and mid in by_id and mid not in sx_ids:
            bad("evalonly rows that disagree with evaluation_only.tsv", f"{mid}: not a second row")
    rule = "second-experiment rows in a training pair or outside the other_experiment rung"
    for p in pairs:
        sides = [m for m in (p["a"], p["b"]) if m in sx_ids]
        if not sides:
            continue
        if p["split"] == "train" or p["rung"] != "other_experiment" or len(sides) == 2:
            bad(rule, f"{p['pair_id']} {p['rung']} {p['split']}")
    if sx or "addon_of" in info:
        dataset_probe(T, a, sx_ids, bad)
    base = a.addon_of or info.get("addon_of", "")
    if base:
        V["rows of the base tables changed or missing"] = base_changed(T, Path(base))
    elif sx:
        bad("rows of the base tables changed or missing", "second-experiment rows, no base named")
    return f"second-experiment rows: {len(sx)} (other lab {n_other}, same lab {n_same})"


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
    oxl = load_reviewed_list(a, info)
    for r in rows:
        mid, x = r["measurement_id"], r["experiment"]
        role = L["role"].get(x)
        if role is None and x in oxl:
            pass                                     # decision 3: the add-on lines judge it
        elif role is None:
            bad("measurements outside ledger", mid)
        elif r["in_eic"] != role:
            bad("rows whose EIC split differs from the ledger", f"{mid} {r['in_eic']} != {role}")
        if role is None and x in oxl:
            pass
        elif r["assay"] == C.CONTROL:
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
        if second_row(A, oxl) or second_row(B, oxl):
            eic = [r for r in (A, B) if not second_row(r, oxl)]
            split = BT.other_experiment_split(eic[0]) if eic else "val"
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
    sx_count = addon_rules(a, T, info, rows, pairs + shared, eval_only, L, bad, V)
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
    print(sx_count)
    n_viol = sum(V.values())
    print(f"check: {'PASS' if not n_viol else f'FAIL ({n_viol} violations)'}")
    return 0 if not n_viol else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", required=True)
    ap.add_argument("--superset-of", default="")
    ap.add_argument("--addon-of", default="", help="base tables; default build_info's addon_of")
    ap.add_argument("--other-experiments", default="",
                    help="$OXL; default build_info's other_experiments")
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
