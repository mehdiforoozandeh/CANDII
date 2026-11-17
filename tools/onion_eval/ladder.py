"""The replicate ladder of D12, complete as far as EIC allows (close-out item E4).

Three subcommands; none edits an input.

pairlist  The layers pair list: a derived copy of an eval set for `layers.py` only (never for scoring).
          `evalset.py` keeps a replicate pair only when a != b, so the eval set holds no pseudoreplicate
          pair; this adds every pseudoreplicate row of pairs.tsv (a == b, split train or val, never blind)
          and drops each sibling rung (anisogenic, other_experiment) whose pairs come from fewer than
          MIN_CELLS cells in the whole list: under PI ruling 30 a populated sibling rung moves the
          cell-type row of its assays onto the cells that hold it, and with one cell that row is empty
          (the 4 anisogenic pairs of EIC are all in cell C09, which has no isogenic pair). Writes the
          pair list (key `layers_pair_list`: base path and sha256, the rule) and a changes TSV.

design    Identifiability of the crossed mixed model per assay, with and without the second
          experiments: the observations `mixed_model.select_observations` takes, the estimability rule
          of `mixed_model.estimability` over cell, experiment, replicate (donor), library and lab, the
          number of levels of each factor and the number of cells in which it varies (a contrast inside
          a cell). Reads measurements.tsv only (no counts).

report    The ladder table from the per-pair values of `layers.py run --pairs-out`. One block per assay
          class, never pooled across assay classes. A rung's increment is ASSAY-MATCHED: for each
          (cell, assay) that holds pairs of the rung and of the rung below it, the mean pair value of the
          rung minus that of the rung below; then the mean over the cell's assays, then over cells.
          (The pooled-by-class rows of layers.tsv subtract cell means taken over different assays on
          the two rungs, so the assay mix of a cell enters the increment; the matched form removes it.)
          The rung below is the nearest populated nested rung for pseudoreplicate .. isogenic, and
          isogenic for every rung above it (PI ruling 30: donor and experiment are siblings over
          isogenic). The four other_experiment pair classes are four rows. A row is estimable when its
          matched pairs number at least MIN_PAIRS from at least MIN_CELLS cells (`layers.py`:94);
          otherwise its value is left empty and its flag says "not estimable" with the counts. Interval:
          95% cluster bootstrap over cells, t form (`layers._interval`). `share` is the increment over
          the class's other_cell_type pair value (the total between two tracks that share only the
          assay), the scale of a mixed-model variance fraction. With --layers it also runs
          `total_variance.telescoping_check` on layers.tsv; with --mixed it lists the mixed-model rows.

CLI:
  python tools/onion_eval/ladder.py pairlist --evalset <json> --tables <dir> --out <json> --changes <tsv>
         [--assays A,B]  (smoke run: keep only these assays)
  python tools/onion_eval/ladder.py design --tables <dir> --out <tsv>
  python tools/onion_eval/ladder.py report --pairs <pair_values.tsv.gz> --out-dir <dir>
         [--layers <layers.tsv>] [--mixed <mixed_model.tsv> ...] [--design <design.tsv>] [--n-boot 1000]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_eval import layers as LY  # noqa: E402

MIN_PAIRS, MIN_CELLS = LY.MIN_PAIRS, LY.MIN_CELLS
SIBLINGS = ("anisogenic", "other_experiment")
NESTED = ("pseudoreplicate", "sequencing", "technical", "isogenic")
# The ladder rows: (D12 rung, pair class or "", layer name, what the increment holds)
LADDER = [
    ("pseudoreplicate", "", "sampling", "Poisson sampling of reads: two read halves of one library"),
    ("sequencing", "", "sequencing", "a second sequencing run of one library, over sampling"),
    ("technical", "", "library", "a second library of one biological sample, over the rung below"),
    ("isogenic", "", "biological replicate", "a second biological replicate of one experiment (same "
     "donor), over the nearest populated rung below; holds sequencing and library when those are empty"),
    ("anisogenic", "", "donor", "a second donor inside one experiment, over isogenic"),
    ("other_experiment", "experiment", "experiment repeat", "a second experiment by the same lab on the "
     "same donor, over isogenic"),
    ("other_experiment", "experiment;lab", "lab", "a second experiment by another lab on the same donor, "
     "over isogenic: the pure lab layer of ruling 30 (holds the experiment repeat too)"),
    ("other_experiment", "experiment;lab;donor", "lab with donor", "a second experiment by another lab "
     "on another donor, over isogenic: lab and donor changed together"),
    ("other_experiment", "experiment;donor", "experiment repeat with donor", "a second experiment by the "
     "same lab on another donor, over isogenic"),
    ("other_cell_type", "", "cell type (with donor and experiment)", "two cell types, over isogenic: "
     "cell type, donor and experiment together (the t305 definition)"),
]
REASON_UNBUILT = ("no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 "
                  "rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage)")
REASON_EMPTY = "no pair in EIC for this assay class"
REPORT_COLUMNS = ["assay_class", "stratum", "band", "rung", "pair_class", "layer", "increment_over",
                  "n_pairs_all", "n_cells_all", "n_pairs", "n_cells", "n_assays", "variance",
                  "layer_increment", "ci_low", "ci_high", "ci_width", "share", "flag", "holds"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _read_tsv(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _write_tsv(rows, cols, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(r.get(c, "")) for c in cols])


def _fmt(v):
    if isinstance(v, float):
        return "" if not math.isfinite(v) else repr(v)
    return str(v)


# ---------------------------------------------------------------------------------------------------
# pairlist

def build_pairlist(ev, pairs_rows, min_cells=MIN_CELLS, assays=None):
    """(pair ids, changes). ev: the eval set dict; pairs_rows: pairs.tsv rows; assays: keep only these
    assays (a smoke run), default all. See the module doc."""
    if assays:
        pairs_rows = [p for p in pairs_rows if p["assay"] in set(assays)]
    by_id = {p["pair_id"]: p for p in pairs_rows}
    base = [i for i in (ev.get("replicate_pairs") or []) if i in by_id]
    have = set(base)
    added = [p["pair_id"] for p in pairs_rows if p["rung"] == "pseudoreplicate" and p["a"] == p["b"]
             and p.get("split") in ("train", "val") and p["pair_id"] not in have]
    ids = base + added
    changes = [dict(by_id[i], role="added", reason="pseudoreplicate pair (a == b) of split train or val")
               for i in added]
    drop = set()
    for rung in SIBLINGS:
        rp = [by_id[i] for i in ids if by_id[i]["rung"] == rung]
        cells = {p["cell"] for p in rp}
        if rp and len(cells) < min_cells:
            drop.update(p["pair_id"] for p in rp)
            for p in rp:
                changes.append(dict(p, role="dropped", reason=f"{rung} rung has {len(rp)} pairs from "
                                    f"{len(cells)} cell(s) < {min_cells}: it would empty the cell-type "
                                    "row of its assays (ruling 30)"))
    blind = [i for i in ids if by_id[i].get("split") == "blind" and i not in drop]
    if blind:
        raise ValueError(f"blind pairs in the layers pair list: {blind[:5]}")
    return [i for i in ids if i not in drop], changes


def cmd_pairlist(args):
    ev_path, tables = Path(args.evalset), Path(args.tables)
    ev = json.loads(ev_path.read_text())
    pairs_rows = _read_tsv(tables / "pairs.tsv")
    ids, changes = build_pairlist(ev, pairs_rows, assays=[x for x in args.assays.split(",") if x] or None)
    out = dict(ev, replicate_pairs=ids, layers_pair_list={
        "base": str(ev_path), "base_sha256": _sha256(ev_path), "tables": str(tables),
        "rule": "ladder.py pairlist: + pseudoreplicate pairs (train, val); - sibling rungs from fewer than "
                f"{MIN_CELLS} cells", "added": sum(c["role"] == "added" for c in changes),
        "dropped": sum(c["role"] == "dropped" for c in changes)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    _write_tsv(changes, ["pair_id", "role", "rung", "assay", "cell", "split", "a", "b", "reason"], args.changes)
    print(f"ladder pairlist: {len(ids)} pairs -> {args.out}; added "
          f"{out['layers_pair_list']['added']}, dropped {out['layers_pair_list']['dropped']}")
    return 0


# ---------------------------------------------------------------------------------------------------
# design

DESIGN_COLUMNS = ["assay", "assay_class", "second_experiments", "n_obs", "n_cells", "factor", "n_levels",
                  "cells_with_contrast", "estimable", "note"]


def design_rows(meas_rows, assay, second_experiments):
    """Identifiability rows of one assay (DESIGN_COLUMNS). See the module doc."""
    from onion_eval import mixed_model as MM
    obs, _ = MM.select_observations(meas_rows, assay, None, (), second_experiments)
    cls = next((m["assay_class"] for m in meas_rows if m["assay"] == assay), "")
    cells = sorted({o["cell"] for o in obs})
    base = dict(assay=assay, assay_class=cls, second_experiments=int(second_experiments), n_obs=len(obs),
                n_cells=len(cells))
    if len(obs) < 3 or len(cells) < 2:
        return [dict(base, factor="all", n_levels="", cells_with_contrast="", estimable=0,
                     note=f"too few observations ({len(obs)}) or cells to fit")]
    factors = list(MM.FACTORS) + ["lab"]
    labels = {f: [o[f] for o in obs] for f in factors}
    est = MM.estimability(labels, order=factors)
    rows = []
    for f in factors:
        within = sum(1 for c in cells if len({o[f] for o in obs if o["cell"] == c}) > 1)
        rows.append(dict(base, factor=f, n_levels=len(set(labels[f])),
                         cells_with_contrast=("" if f == "cell" else within), estimable=est[f][0],
                         note=est[f][1]))
    return rows


def cmd_design(args):
    meas = _read_tsv(Path(args.tables) / "measurements.tsv")
    assays = sorted({m["assay"] for m in meas if m.get("assay_class") != "control"})
    rows = [r for a in assays for se in (False, True) for r in design_rows(meas, a, se)]
    _write_tsv(rows, DESIGN_COLUMNS, args.out)
    print(f"ladder design: {len(rows)} rows -> {args.out}")
    return 0


# ---------------------------------------------------------------------------------------------------
# report

def unit_values(pair_rows):
    """{(assay_class, stratum, band): {(rung, pair_class): {(cell, assay): [pair values]}}}; a pair value
    is 0.5 var_d (the `layers.py` pair value)."""
    out = {}
    for r in pair_rows:
        if r["var_d"] == "":
            continue
        key = (r["assay_class"], r["stratum"], r["band"])
        rk = (r["rung"], r["pair_class"] if r["rung"] == "other_experiment" else "")
        out.setdefault(key, {}).setdefault(rk, {}).setdefault((r["cell"], r["assay"]), []).append(
            0.5 * float(r["var_d"]))
    return out


def _lower(rung, populated):
    """The (rung, "") whose pair value the ladder row subtracts, or None (innermost populated rung)."""
    if rung in NESTED:
        prev = [q for q in NESTED[:NESTED.index(rung)] if q in populated]
        return (prev[-1], "") if prev else None
    return ("isogenic", "")


def ladder_block(units, n_boot=1000, seed=0):
    """Ladder rows of one (assay class, stratum, band). units: the value of `unit_values` for it."""
    populated = {rk[0] for rk, u in units.items() if u}
    oct_ = units.get(("other_cell_type", ""), {})
    oct_cells = {}
    for (c, _a), v in oct_.items():
        oct_cells.setdefault(c, []).extend(v)
    total = float(np.mean([np.mean(v) for v in oct_cells.values()])) if oct_cells else float("nan")
    rows = []
    for rung, pc, layer, holds in LADDER:
        u = units.get((rung, pc), {})
        n_all = sum(len(v) for v in u.values())
        c_all = len({c for c, _ in u})
        low = _lower(rung, populated)
        row = dict(rung=rung, pair_class=pc, layer=layer, holds=holds, n_pairs_all=n_all, n_cells_all=c_all,
                   increment_over=low[0] if low else "", n_pairs=0, n_cells=0, n_assays=0,
                   variance=float("nan"), layer_increment=float("nan"), ci_low=float("nan"),
                   ci_high=float("nan"), ci_width=float("nan"), share=float("nan"))
        if not u:
            row["flag"] = REASON_UNBUILT if rung in ("sequencing", "technical") else REASON_EMPTY
            rows.append(row)
            continue
        lu = units.get(low, {}) if low else None
        matched = sorted(u) if low is None else sorted(k for k in u if k in lu)
        n_m = sum(len(u[k]) for k in matched)
        cells = sorted({c for c, _ in matched})
        row.update(n_pairs=n_m, n_cells=len(cells), n_assays=len({a for _, a in matched}))
        if n_m < MIN_PAIRS or len(cells) < MIN_CELLS:
            below = "" if low is None else f" with a {low[0]} pair in the same cell and assay"
            row["flag"] = (f"not estimable: {n_m} pairs from {len(cells)} cells{below} "
                           f"(rule: {MIN_PAIRS} pairs from {MIN_CELLS} cells); all pairs of the row: "
                           f"{n_all} from {c_all} cells")
            rows.append(row)
            continue
        inc_c, var_c = {}, {}
        for c in cells:
            ks = [k for k in matched if k[0] == c]
            var_c[c] = float(np.mean([np.mean(u[k]) for k in ks]))
            inc_c[c] = float(np.mean([np.mean(u[k]) - (np.mean(lu[k]) if low else 0.0) for k in ks]))
        vals = np.array([inc_c[c] for c in cells])
        est = float(vals.mean())
        rng = np.random.default_rng(seed)
        boot = vals[rng.integers(0, len(cells), size=(n_boot, len(cells)))].mean(1)
        lo, hi = LY._interval(est, boot, len(cells))
        row.update(variance=float(np.mean([var_c[c] for c in cells])), layer_increment=est, ci_low=lo,
                   ci_high=hi, ci_width=hi - lo, share=est / total if total and math.isfinite(total) else
                   float("nan"), flag="ok")
        rows.append(row)
    return rows


def ladder_rows(pair_rows, n_boot=1000, seed=0):
    out = []
    for (cls, s, b), units in sorted(unit_values(pair_rows).items()):
        if cls == "control":
            continue
        for r in ladder_block(units, n_boot, seed):
            out.append(dict(assay_class=cls, stratum=s, band=b, **r))
    return out


def _g(x, nd=3):
    return "" if not (isinstance(x, float) and math.isfinite(x)) else f"{x:.{nd}g}"


def markdown(rows, tele=None, mixed=None, design=None):
    lines = ["# Replicate ladder (close-out item E4)", "",
             "Stratum all, band all. Increments are assay-matched pair values (0.5 E[(Y - Y')^2] of log1p "
             "counts at each assay's common depth); intervals are 95% cluster-bootstrap intervals over cells. "
             "No row is pooled across assay classes. Full table: ladder.tsv.", ""]
    for cls in sorted({r["assay_class"] for r in rows}):
        lines += [f"## {cls}", "", "| rung | layer | increment [95% interval] | pairs (cells) matched | "
                  "pairs (cells) all | share of cell-type pair value | flag |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["assay_class"] != cls or r["stratum"] != "all" or r["band"] != "all":
                continue
            rung = r["rung"] + (f" ({r['pair_class']})" if r["pair_class"] else "")
            val = (f"{_g(r['layer_increment'], 4)} [{_g(r['ci_low'], 3)}, {_g(r['ci_high'], 3)}]"
                   if r["flag"] == "ok" else "")
            lines.append(f"| {rung} | {r['layer']} | {val} | {r['n_pairs']} ({r['n_cells']}) | "
                         f"{r['n_pairs_all']} ({r['n_cells_all']}) | {_g(r['share'], 3)} | {r['flag']} |")
        lines.append("")
    if tele is not None:
        ok = sum(1 for t in tele if t["ok"])
        lines += ["## Telescoping check of layers.tsv", "",
                  f"{ok} of {len(tele)} groups telescope exactly (total_variance.telescoping_check); "
                  f"{sum(1 for t in tele if t['chain_ok'])} have the ruling-30 chain. The rest hold rungs with "
                  "different cells (telescoping.tsv).", ""]
    if mixed:
        lines += ["## Crossed mixed model (order-free cross-check)", "",
                  "| assay | band | factor | variance fraction [95% interval] | levels | estimable | note |",
                  "|---|---|---|---|---|---|---|"]
        for m in mixed:
            frac = (f"{_g(float(m['variance_fraction']), 3)} [{_g(float(m['ci_low']), 3)}, "
                    f"{_g(float(m['ci_high']), 3)}]" if m["variance_fraction"] not in ("", None) else "")
            lines.append(f"| {m['assay']} | {m['band']} | {m['factor']} | {frac} | {m['n_levels']} | "
                         f"{m['estimable']} | {m['note']} |")
        lines.append("")
    if design:
        lines += ["## Identifiability of the mixed model (design.tsv)", "",
                  "| assay | second experiments | obs | cells | experiment | replicate (donor) | library | lab |",
                  "|---|---|---|---|---|---|---|---|"]
        by = {}
        for d in design:
            by.setdefault((d["assay"], d["second_experiments"], d["n_obs"], d["n_cells"]), {})[d["factor"]] = d
        for (a, se, n, c), fs in sorted(by.items()):
            def cell(f):
                d = fs.get(f)
                return "" if d is None else f"{'yes' if d['estimable'] == '1' else 'no'} ({d['cells_with_contrast']})"
            lines.append(f"| {a} | {se} | {n} | {c} | {cell('experiment')} | {cell('replicate')} | "
                         f"{cell('library')} | {cell('lab')} |")
        lines += ["", "Cell entries: estimable by mixed_model.estimability (cells in which the factor varies).", ""]
    return "\n".join(lines) + "\n"


def cmd_report(args):
    pair_rows = _read_tsv(args.pairs)
    rows = ladder_rows(pair_rows, args.n_boot, args.seed)
    out = Path(args.out_dir)
    _write_tsv(rows, REPORT_COLUMNS, out / "ladder.tsv")
    tele = None
    if args.layers:
        from onion_eval import total_variance as TV
        tele = TV.telescoping_check(args.layers)
        _write_tsv(tele, ["group", "innermost", "outermost", "sum_of_increments", "outermost_variance",
                          "abs_diff", "chain_ok", "ok"], out / "telescoping.tsv")
    mixed = [r for p in (args.mixed or []) for r in _read_tsv(p)]
    design = _read_tsv(args.design) if args.design else None
    (out / "ladder.md").write_text(markdown(rows, tele, mixed, design))
    print(f"ladder report: {len(rows)} rows -> {out / 'ladder.tsv'}, {out / 'ladder.md'}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairlist")
    p.add_argument("--evalset", required=True)
    p.add_argument("--tables", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--changes", required=True)
    p.add_argument("--assays", default="", help="comma list: keep only these assays (smoke run)")
    d = sub.add_parser("design")
    d.add_argument("--tables", required=True)
    d.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--pairs", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--layers", default=None)
    r.add_argument("--mixed", nargs="*", default=None)
    r.add_argument("--design", default=None)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    return {"pairlist": cmd_pairlist, "design": cmd_design, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
