"""Pilot tables over the existing EIC store (t152 chunk P1, crux task t165).

Writes a tables directory in the pinned interface B layout that exposes `CANDI_STORE/eic` on Fir
through `store:` counts paths, so the miniCANDI loader, the six rungs and the scorer can run end to
end before the new EIC dataset exists. Plumbing only: it settles no ablation.

  python tools/onion_data/pilot_tables.py build --corpus <CANDI_STORE/eic> --out <tables dir>
  python tools/onion_data/pilot_tables.py check --tables <tables dir>

What `build` writes (`measurements.tsv`, `pairs.tsv`, `withheld.tsv`, `covariates.json`):
  roles      `in_eic` from the store's biosample prefix: `T_` train, `V_` val, `B_` blind; the cell
             is the name after the prefix, so one cell holds tracks of several roles (EIC splits
             per track)
  tracks     every store track, the 35 assays and `chipseq-control` (controls kept);
             measurement_id = `<exp_accession>.pooled.native.base`, layer = native,
             counts_path = `store:<corpus root>:<biosample>:<assay>`
  shared     a control file that two biosamples of one cell share (same cell, assay and file
             accession; 20 such pairs on 2026-11-10) is ONE measurement: it keeps the least
             restricted role (train < val < blind) and that biosample's store path
  depth arms the only counterfactual the store offers: read-free binomial thinning of each base
             track (`arm_knob = depth`, `arm_value` = the fraction kept, `reads_subset_of` = the
             base); OnionDataset thins `store:` rows with `arm_knob = depth` on read
  pairs      `other_cell_type` pairs (same assay, two cells, base views) and `designed` pairs
             (base vs each depth arm, `subset_pair = 1`); `split` is the more restricted role of
             the two sides; `never_trained = 0` everywhere
  withheld   header only
  covariates vocabularies of lab and platform from the store metadata (pipeline, antibody and
             donor are unknown in the store and stay empty, id 0)
It also links `<out>/../../genome` to the store's genome layer (`<corpus>/../genome`) when absent.

Not in the store, left empty: eic_track (the D3 ledger maps it), bio_rep, tech_rep, library,
depth_fragments, pipeline, pipeline_version, dedup, mapq_threshold, antibody, donor, QC columns,
sha256. has_pseudoreps = has_signal = 0 (no reads, no signal.h5).

`check` validates the tables against the schema and the store manifest, prints the store tracks per
role (`train 267+40 controls, val 45+24 controls, blind 51+9 controls` on Fir), reads one bin of
every counts_path, and iterates real training and validation batches through `OnionDataset`
(validation from a temporary evalset), printing their shapes and the seconds per training batch.
Exit 0 when every check passes, else 1.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import contract as C  # noqa: E402
from minicandi.dataset import (MEASUREMENT_COLUMNS, PAIR_COLUMNS, WITHHELD_COLUMNS,  # noqa: E402
                               read_tsv)

PREFIX_ROLE = {"T_": "train", "V_": "val", "B_": "blind"}
ROLE_ORDER = {"train": 0, "val": 1, "blind": 2}
RUNGS = ("pseudoreplicate", "sequencing", "technical", "isogenic", "anisogenic",
         "other_experiment", "other_cell_type", "designed")
DEFAULT_DEPTH_FRACTIONS = (0.5, 0.25)
VOCAB_FIELDS = ("pipeline", "pipeline_version", "lab", "platform", "antibody", "donor")


def _s(v) -> str:
    return "" if v is None else str(v)


def _int_str(v) -> str:
    if v is None or str(v).strip() == "":
        return ""
    return str(int(round(float(v))))


def store_path(corpus: Path, bios: str, assay: str) -> str:
    return f"store:{corpus}:{bios}:{assay}"


def parse_store_path(p: str) -> tuple[str, str, str]:
    if not p.startswith("store:"):
        raise ValueError(f"not a store: path: {p!r}")
    root, bios, assay = p[len("store:"):].rsplit(":", 2)
    return root, bios, assay


def role_of(bios: str) -> str:
    role = PREFIX_ROLE.get(bios[:2])
    if role is None:
        raise ValueError(f"biosample {bios!r}: prefix not in {sorted(PREFIX_ROLE)}")
    return role


def load_manifest(corpus: Path) -> dict:
    return json.loads((Path(corpus) / "manifest.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------- build

def build_rows(corpus: Path, depth_fractions=DEFAULT_DEPTH_FRACTIONS) -> tuple[list, list, dict]:
    """(measurement rows, pair rows, sharing report) from the store manifest."""
    corpus = Path(corpus).resolve()
    man = load_manifest(corpus)
    vocab = list(man["assay_vocabulary"])
    bad = [a for a in vocab if a not in C.TRACK_INDEX]
    if bad:
        raise ValueError(f"store assays not in the 36-track panel: {bad}")

    # one measurement per (cell, assay, file accession); least restricted role wins
    chosen: dict[tuple, tuple] = {}
    sources: dict[tuple, list] = defaultdict(list)
    for bios in sorted(man["biosamples"]):
        role = role_of(bios)
        cell = bios[2:]
        for t in man["biosamples"][bios]["tracks"]:
            assay = t["assay"]
            if assay not in C.TRACK_INDEX:
                raise ValueError(f"{bios}/{assay}: not in the 36-track panel")
            key = (cell, assay, _s(t.get("file_accession")) or f"{bios}/{assay}")
            sources[key].append(bios)
            if key not in chosen or ROLE_ORDER[role] < ROLE_ORDER[chosen[key][1]]:
                chosen[key] = (bios, role, t)

    base_of_bios: dict[tuple, str] = {}       # (biosample, assay) -> base measurement id
    rows, by_id = [], {}
    for key in sorted(chosen, key=lambda k: (k[0], C.TRACK_INDEX[k[1]], k[2])):
        bios, role, t = chosen[key]
        cell, assay, _ = key
        exp = _s(t.get("exp_accession"))
        if not exp:
            raise ValueError(f"{bios}/{assay}: no exp_accession in the manifest")
        mid = f"{exp}.pooled.native.base"
        if mid in by_id:
            raise ValueError(f"measurement id {mid} from two different store tracks "
                             f"({by_id[mid]['counts_path']} and {bios}/{assay})")
        row = {k: "" for k in MEASUREMENT_COLUMNS}
        row.update({
            "measurement_id": mid, "experiment": exp, "assay": assay,
            "assay_class": C.assay_class(assay), "cell": cell, "in_eic": role,
            "unit": "pooled", "layer": "native",
            "depth_reads": _int_str(t.get("depth")), "run_type": _s(t.get("run_type")),
            "read_length": _int_str(t.get("read_length")), "platform": _s(t.get("platform")),
            "lab": _s(t.get("lab")), "has_pseudoreps": "0", "has_signal": "0",
            "counts_path": store_path(corpus, bios, assay),
        })
        rows.append(row)
        by_id[mid] = row
        for b in sources[key]:
            base_of_bios[(b, assay)] = mid

    # control of each non-control row: the control of the biosample it was read from
    for row in rows:
        if row["assay"] == C.CONTROL:
            continue
        _, bios, _ = parse_store_path(row["counts_path"])
        row["control_measurement_id"] = base_of_bios.get((bios, C.CONTROL), "")

    base_rows = list(rows)
    pairs = []

    def add_pair(a, b, rung, knobs, subset, split):
        pairs.append({"pair_id": f"p{len(pairs) + 1:06d}", "a": a["measurement_id"],
                      "b": b["measurement_id"], "rung": rung, "assay": a["assay"],
                      "cell": a["cell"] if a["cell"] == b["cell"] else "",
                      "knobs_differing": knobs, "subset_pair": str(subset), "split": split,
                      "never_trained": "0"})

    # designed depth arms (subset pairs)
    for base in base_rows:
        for f in depth_fractions:
            f = float(f)
            if not 0.0 < f < 1.0:
                raise ValueError(f"depth fraction {f} not in (0, 1)")
            arm = dict(base)
            arm.update({
                "measurement_id": base["measurement_id"].replace(".base", f".depth{f:.2f}"),
                "arm_knob": "depth", "arm_value": f"{f:.2f}",
                "parent_measurement_id": base["measurement_id"],
                "reads_subset_of": base["measurement_id"],
                "depth_reads": (str(int(round(float(base["depth_reads"]) * f)))
                                if base["depth_reads"] else ""),
            })
            rows.append(arm)
            add_pair(base, arm, "designed", "depth", 1, base["in_eic"])

    # other_cell_type pairs among base views of one assay
    by_assay = defaultdict(list)
    for r in base_rows:
        by_assay[r["assay"]].append(r)
    for assay in sorted(by_assay, key=C.TRACK_INDEX.get):
        for a, b in combinations(by_assay[assay], 2):
            if a["cell"] == b["cell"]:
                continue
            split = max(a["in_eic"], b["in_eic"], key=ROLE_ORDER.get)
            add_pair(a, b, "other_cell_type", "cell", 0, split)

    shared = {k: v for k, v in sources.items() if len(v) > 1}
    return rows, pairs, {"shared": shared, "manifest": man}


def write_tsv(path: Path, columns, rows) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def covariates_spec(rows) -> dict:
    vocab = {f: sorted({r[f] for r in rows if r[f]}) for f in VOCAB_FIELDS}
    return {"schema": 1, "meta_columns": list(C.META_COLUMNS), "qc_columns": list(C.QC_COLUMNS),
            "vocab": vocab}


def link_genome(corpus: Path, out: Path) -> Path:
    """`<out>/../../genome` -> `<corpus>/../genome`; an existing link must resolve to it."""
    target = (Path(corpus).resolve().parent / "genome")
    link = Path(out).parent.parent / "genome"
    if link.is_symlink() or link.exists():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"{link} resolves to {link.resolve()}, want {target}")
        return link
    link.symlink_to(target)
    return link


def cmd_build(args) -> int:
    corpus = Path(args.corpus).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, pairs, info = build_rows(corpus, args.depth_fractions)
    write_tsv(out / "measurements.tsv", MEASUREMENT_COLUMNS, rows)
    write_tsv(out / "pairs.tsv", PAIR_COLUMNS, pairs)
    write_tsv(out / "withheld.tsv", WITHHELD_COLUMNS, [])
    spec = covariates_spec(rows)
    (out / "covariates.json").write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    link = link_genome(corpus, out)
    n_base = sum(1 for r in rows if not r["arm_knob"])
    print(f"wrote {out}: {len(rows)} measurements ({n_base} base, {len(rows) - n_base} depth "
          f"arms), {len(pairs)} pairs, 0 withheld")
    print(f"genome link {link} -> {link.resolve()}")
    for (cell, assay, facc), bios in sorted(info["shared"].items()):
        print(f"shared file {facc} ({cell} {assay}): biosamples {','.join(bios)} -> one "
              f"measurement")
    return 0


# ------------------------------------------------------------------------------------- check

class Report:
    def __init__(self):
        self.failures = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"FAIL {msg}")


def store_track_counts(rows, rep: Report) -> dict:
    """Store tracks per role (from each referenced manifest), each mapped to exactly one row."""
    roots = sorted({parse_store_path(r["counts_path"])[0] for r in rows
                    if r["counts_path"].startswith("store:")})
    base = {(r["cell"], r["assay"]): [] for r in rows}
    for r in rows:
        if not r["arm_knob"]:
            base[(r["cell"], r["assay"])].append(r)
    counts = Counter()
    for root in roots:
        man = load_manifest(Path(root))
        for bios, b in man["biosamples"].items():
            for t in b["tracks"]:
                ctl = t["assay"] == C.CONTROL
                counts[(role_of(bios), ctl)] += 1
                cands = base.get((bios[2:], t["assay"]), [])
                hits = [r for r in cands if r["experiment"] == _s(t.get("exp_accession"))]
                if len(hits) != 1:
                    rep.fail(f"store track {bios}/{t['assay']} maps to {len(hits)} rows")
                    continue
                r = hits[0]
                if ROLE_ORDER[r["in_eic"]] > ROLE_ORDER[role_of(bios)]:
                    rep.fail(f"{r['measurement_id']}: role {r['in_eic']} more restricted than "
                             f"its store biosample {bios}")
                if r["depth_reads"] != _int_str(t.get("depth")):
                    rep.fail(f"{r['measurement_id']}: depth_reads {r['depth_reads']} != "
                             f"manifest {t.get('depth')}")
    return counts


def check_readable(rows, rep: Report) -> int:
    from candi.store.reader import CorpusStore
    stores, n = {}, 0
    for r in rows:
        if r["arm_knob"]:
            continue
        try:
            root, bios, assay = parse_store_path(r["counts_path"])
            cs = stores.get(root) or stores.setdefault(root, CorpusStore(root))
            bs = cs[bios]
            chrom = bs.chroms()[0]
            v = bs.control(chrom, 0, 1)[:, 0] if assay == C.CONTROL else \
                bs[assay].counts(chrom, 0, 1)
            if v.shape != (1,):
                raise ValueError(f"shape {v.shape}")
            n += 1
        except Exception as e:  # noqa: BLE001 - report every failure
            rep.fail(f"{r['measurement_id']}: counts_path {r['counts_path']} unreadable: {e}")
    for cs in stores.values():
        cs.close()
    return n


def temp_evalset(rows, path: Path, n_targets: int, context_bins: int, chrom: str,
                 start_bin: int) -> Path:
    """A small evalset.json: the first validation base tracks, inputs = the cell's training base
    tracks (pinned interface D shape; E6 builds the real one)."""
    train_base = defaultdict(list)
    for r in rows:
        if r["in_eic"] == "train" and not r["arm_knob"]:
            train_base[r["cell"]].append(r["measurement_id"])
    targets = []
    for r in rows:
        if r["in_eic"] == "val" and not r["arm_knob"] and r["assay"] != C.CONTROL:
            # EIC splits per track: the cell's training tracks never hold the target's assay
            targets.append({"target": r["measurement_id"],
                            "inputs": sorted(train_base.get(r["cell"], [])),
                            "regime": "leave_one_assay_out", "split": "val"})
        if len(targets) >= n_targets:
            break
    ev = {"tables": "pilot_check", "regions": [[chrom, int(start_bin)]],
          "imputation_targets": targets, "correction_pairs": [], "replicate_pairs": [],
          "scorable": {}, "strata": []}
    path.write_text(json.dumps(ev), encoding="utf-8")
    return path


def shapes(batch) -> str:
    return ", ".join(f"{k} {tuple(v.shape)}" for k, v in batch.items() if k != "meta")


def check_dataset(tables: Path, rows, args, rep: Report) -> None:
    from minicandi.dataset import OnionDataset, RoleError
    by_id = {r["measurement_id"]: r for r in rows}
    cfg = {"context_bins": args.context_bins, "batch_size": args.batch_size}
    ds = OnionDataset(tables, "train", cfg, seed=args.seed)
    times = []
    for i in range(args.batches):
        t0 = time.perf_counter()
        b = ds.batch(args.batch_size)
        times.append(time.perf_counter() - t0)
        ids = {m for meta in b["meta"] for m in meta["inputs"] + meta["targets"]}
        wrong = sorted(m for m in ids if by_id[m]["in_eic"] != "train")
        if wrong:
            rep.fail(f"training batch {i} read non-training rows {wrong[:5]}")
        if b["x_counts"].shape[-1] != C.A or b["c_out"].shape[1] != C.A:
            rep.fail(f"training batch {i}: track axis is not {C.A}")
        kinds = Counter(int(k) for k in b["y_kind"][b["y_avail"]].flatten())
        print(f"train batch {i}: {shapes(b)}; cells {[m['cell'] for m in b['meta']]}; "
              f"y_kind counts {dict(sorted(kinds.items()))}; {times[-1]:.2f} s")
    later = times[1:] or times
    print(f"seconds per training batch (batch_size {args.batch_size}, context_bins "
          f"{args.context_bins}): first {times[0]:.2f}, mean of the rest "
          f"{sum(later) / len(later):.2f}")

    with tempfile.TemporaryDirectory() as td:
        chrom = ds.chroms[0]
        ev = temp_evalset(rows, Path(td) / "evalset.json", args.val_targets, args.context_bins,
                          chrom, min(ds.n_bins[chrom] // 2, ds.n_bins[chrom] - args.context_bins))
        vcfg = dict(cfg, evalset=str(ev), region_bins=args.context_bins)
        vds = OnionDataset(tables, "val", vcfg, seed=args.seed)
        n = 0
        for i, b in enumerate(vds):
            ids = {m for meta in b["meta"] for m in meta["inputs"] + meta["targets"]}
            blind = sorted(m for m in ids if by_id[m]["in_eic"] == "blind")
            if blind:
                rep.fail(f"validation batch {i} read blind rows {blind[:5]}")
            tasks = Counter(m.get("task") for m in b["meta"])
            print(f"val batch {i}: {shapes(b)}; tasks {dict(tasks)}")
            n += 1
            if n >= args.batches:
                break
        if n == 0:
            rep.fail("no validation batch")
    try:
        OnionDataset(tables, "blind", cfg)
        rep.fail("mode blind opened without allow_blind")
    except RoleError:
        print("mode blind without allow_blind: refused")


def cmd_check(args) -> int:
    tables = Path(args.tables)
    rep = Report()
    rows = read_tsv(tables / "measurements.tsv", MEASUREMENT_COLUMNS)
    pairs = read_tsv(tables / "pairs.tsv", PAIR_COLUMNS)
    withheld = read_tsv(tables / "withheld.tsv", WITHHELD_COLUMNS)
    spec = json.loads((tables / "covariates.json").read_text(encoding="utf-8"))
    if tuple(spec.get("meta_columns", ())) != C.META_COLUMNS:
        rep.fail("covariates.json meta_columns differ from the contract")
    if withheld:
        rep.fail(f"withheld.tsv has {len(withheld)} rows, want 0")
    by_id = {}
    for r in rows:
        if r["measurement_id"] in by_id:
            rep.fail(f"duplicate measurement_id {r['measurement_id']}")
        by_id[r["measurement_id"]] = r
        if r["in_eic"] not in ROLE_ORDER:
            rep.fail(f"{r['measurement_id']}: in_eic {r['in_eic']!r}")
        if r["layer"] != "native" or not r["counts_path"].startswith("store:"):
            rep.fail(f"{r['measurement_id']}: layer {r['layer']} / path {r['counts_path']}")
        if r["arm_knob"] not in ("", "depth"):
            rep.fail(f"{r['measurement_id']}: arm_knob {r['arm_knob']!r}")
        if r["arm_knob"] == "depth":
            p = by_id.get(r["reads_subset_of"])
            if p is None or p["counts_path"] != r["counts_path"] or p["in_eic"] != r["in_eic"]:
                rep.fail(f"{r['measurement_id']}: reads_subset_of {r['reads_subset_of']!r} is "
                         f"not its base")
    base = [r for r in rows if not r["arm_knob"]]

    tracks = {r["assay"] for r in rows}
    missing = [t for t in C.TRACKS if t not in tracks]
    print(f"channels: {len(tracks)} of {C.A} (35 assays + {C.CONTROL})"
          + (f"; missing {missing}" if missing else ""))
    if missing:
        rep.fail(f"tracks missing from the tables: {missing}")

    counts = store_track_counts(rows, rep)
    print("store tracks: " + ", ".join(
        f"{role} {counts[(role, False)]}+{counts[(role, True)]} controls" for role in ROLE_ORDER))
    mc = Counter((r["in_eic"], r["assay"] == C.CONTROL) for r in base)
    print("measurements (base views): " + ", ".join(
        f"{role} {mc[(role, False)]}+{mc[(role, True)]} controls" for role in ROLE_ORDER)
        + f"; depth-arm rows {len(rows) - len(base)}; total {len(rows)}")

    pc = Counter()
    seen = set()
    for p in pairs:
        if p["pair_id"] in seen:
            rep.fail(f"duplicate pair_id {p['pair_id']}")
        seen.add(p["pair_id"])
        a, b = by_id.get(p["a"]), by_id.get(p["b"])
        if a is None or b is None:
            rep.fail(f"{p['pair_id']}: unknown measurement")
            continue
        split = max(a["in_eic"], b["in_eic"], key=ROLE_ORDER.get)
        if p["rung"] not in RUNGS or p["split"] != split or p["never_trained"] != "0":
            rep.fail(f"{p['pair_id']}: rung {p['rung']} split {p['split']} (want {split}) "
                     f"never_trained {p['never_trained']}")
        if p["rung"] == "other_cell_type":
            ok = a["assay"] == b["assay"] and a["cell"] != b["cell"] and p["subset_pair"] == "0"
        elif p["rung"] == "designed":
            ok = (b["reads_subset_of"] == a["measurement_id"] and p["subset_pair"] == "1"
                  and p["knobs_differing"] == "depth")
        else:
            ok = False
        if not ok:
            rep.fail(f"{p['pair_id']}: {p['rung']} pair is inconsistent")
        pc[(p["rung"], p["split"], p["subset_pair"])] += 1
    for (rung, split, sub), n in sorted(pc.items()):
        print(f"pairs {rung} {split} subset_pair={sub}: {n}")

    n_ok = check_readable(rows, rep)
    if n_ok == len(base):
        print(f"every counts_path readable ({n_ok} store tracks)")

    if not args.no_dataset:
        try:
            check_dataset(tables, rows, args, rep)
        except Exception as e:  # noqa: BLE001
            rep.fail(f"OnionDataset: {type(e).__name__}: {e}")
    print(f"check: {'PASS' if not rep.failures else f'FAIL ({len(rep.failures)} failures)'}")
    return 0 if not rep.failures else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the pilot tables from a CANDI_STORE corpus")
    b.add_argument("--corpus", required=True, help="corpus root, e.g. .../CANDI_STORE/eic")
    b.add_argument("--out", required=True, help="tables directory to write")
    b.add_argument("--depth-fractions", type=float, nargs="*",
                   default=list(DEFAULT_DEPTH_FRACTIONS),
                   help="fractions kept by the read-free thinning arms (default 0.5 0.25)")
    b.set_defaults(func=cmd_build)
    c = sub.add_parser("check", help="validate pilot tables and iterate OnionDataset batches")
    c.add_argument("--tables", required=True)
    c.add_argument("--batches", type=int, default=4)
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--context-bins", type=int, default=1024)
    c.add_argument("--val-targets", type=int, default=4)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--no-dataset", action="store_true", help="skip the OnionDataset batches")
    c.set_defaults(func=cmd_check)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
