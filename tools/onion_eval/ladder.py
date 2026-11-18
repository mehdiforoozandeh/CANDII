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
          matched pairs number at least MIN_PAIRS from at least MIN_CELLS cells (`layers.MIN_PAIRS`, `layers.MIN_CELLS`, layers.py:100);
          otherwise its value is left empty and its flag says "not estimable" with the counts. Interval:
          95% cluster bootstrap over cells, t form (`layers._interval`). A negative increment is kept
          (never clipped) and flagged. `share` is, on the row's (cell, assay) units that also hold
          other_cell_type pairs, the increment over the other_cell_type pair value of those same units
          (the total between two tracks that share only the assay; the scale of a mixed-model variance
          fraction); `share_units` counts them. With --mixed it lists the mixed-model rows; with --design
          it applies the quote rule (a factor other than cell that varies inside fewer than 3 cells:
          fraction left empty, "not identified inside cells").

          Two forms (PI decision of 2026-11-17, row 5: two-sided is the ladder definition).
          two_sided (default; needs --tables for the experiment of each side of a pair): the rungs whose
            two sides are two different experiments (TWO_SIDED: other_cell_type, other_experiment)
            subtract, pair by pair, 0.5 x (L(a) + L(b)), where L(x) is the mean isogenic pair value of
            the experiment of side x in the same assay, stratum and band. A pair whose a-side or b-side
            experiment has no isogenic pair leaves the matched set; `n_pairs_left_b` counts those that
            left only because of the b-side. A (cell, assay) unit is kept while it holds a kept pair.
            The bootstrap resamples all cells jointly (`_joint_boot`): a drawn cell counts wherever it
            appears, as a-side or b-side, and L is recomputed inside each draw. The nested rungs and anisogenic
            (both sides in one experiment) are the same in both forms.
          one_sided (`--form one_sided`; tonight's form, kept for comparison): every rung subtracts the
            lower rung of the a-side unit only. Writes ladder.tsv byte-identical to the earlier code, the
            ladder's own telescoping check `chain_check.tsv`, and with --layers the old
            `total_variance.telescoping_check` (telescoping.tsv).
          Both forms write `common_cells.tsv`, the acceptance check that replaces telescoping: per
          (assay class, stratum, band), the rungs pseudoreplicate, isogenic and other_cell_type are
          recomputed on the a-side cells that every one of them holds; the check passes when there are at
          least MIN_CELLS common cells and each common-cell increment lies inside its full row's interval.

CLI:
  python tools/onion_eval/ladder.py pairlist --evalset <json> --tables <dir> --out <json> --changes <tsv>
         [--assays A,B]  (smoke run: keep only these assays)
  python tools/onion_eval/ladder.py design --tables <dir> --out <tsv>
  python tools/onion_eval/ladder.py report --pairs <pair_values.tsv.gz> --out-dir <dir>
         [--form two_sided|one_sided] [--tables <dir with pairs.tsv, measurements.tsv>]
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
TWO_SIDED = ("other_cell_type", "other_experiment")     # the two sides are two different experiments
FORMS = ("two_sided", "one_sided")
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
                  "layer_increment", "ci_low", "ci_high", "ci_width", "share", "share_units", "flag", "holds"]


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

def unit_values(pair_rows, with_ids=False):
    """{(assay_class, stratum, band): {(rung, pair_class): {(cell, assay): [pair values]}}}; a pair value
    is 0.5 var_d (the `layers.py` pair value). with_ids: each entry is (pair_id, pair value)."""
    out = {}
    for r in pair_rows:
        if r["var_d"] == "":
            continue
        key = (r["assay_class"], r["stratum"], r["band"])
        rk = (r["rung"], r["pair_class"] if r["rung"] == "other_experiment" else "")
        v = 0.5 * float(r["var_d"])
        out.setdefault(key, {}).setdefault(rk, {}).setdefault((r["cell"], r["assay"]), []).append(
            (r["pair_id"], v) if with_ids else v)
    return out


def pair_sides(pairs_rows, meas_rows):
    """{pair_id: (experiment of side a, experiment of side b, cell of side a, cell of side b)} from
    pairs.tsv and measurements.tsv. The cells drive the joint cluster bootstrap of the two-sided form."""
    m = {r["measurement_id"]: (r["experiment"], r["cell"]) for r in meas_rows}
    return {p["pair_id"]: (m[p["a"]][0], m[p["b"]][0], m[p["a"]][1], m[p["b"]][1]) for p in pairs_rows}


def _lower_by_experiment(lower_ids, sides):
    """{(experiment, assay): mean lower-rung pair value} from (cell, assay) -> [(pair_id, value)]. A
    lower-rung pair (pseudoreplicate, isogenic) has both sides in one experiment: its a-side's."""
    acc = {}
    for (_, a), lst in lower_ids.items():
        for pid, v in lst:
            acc.setdefault((sides[pid][0], a), []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _two_sided_kept(u_ids, low_by_exp, sides):
    """(kept, n_left_a, n_left_b). kept: {(cell, assay): [(pair value, pair value - 0.5 (L(a) + L(b)),
    pair_id)]}.
    A pair leaves when its a-side experiment has no lower value (n_left_a), else when its b-side
    experiment has none (n_left_b)."""
    kept, n_a, n_b = {}, 0, 0
    for k, lst in u_ids.items():
        for pid, v in lst:
            ea, eb = sides[pid][0], sides[pid][1]
            la, lb = low_by_exp.get((ea, k[1])), low_by_exp.get((eb, k[1]))
            if la is None:
                n_a += 1
            elif lb is None:
                n_b += 1
            else:
                kept.setdefault(k, []).append((v, v - 0.5 * (la + lb), pid))
    return kept, n_a, n_b


def _lower(rung, populated):
    """The (rung, "") whose pair value the ladder row subtracts, or None (innermost populated rung)."""
    if rung in NESTED:
        prev = [q for q in NESTED[:NESTED.index(rung)] if q in populated]
        return (prev[-1], "") if prev else None
    return ("isogenic", "")


def ladder_block(units, n_boot=1000, seed=0, sides=None):
    """Ladder rows of one (assay class, stratum, band). units: the value of `unit_values` for it.
    sides None: the one-sided form (units hold values). sides given: the two-sided form (units hold
    (pair_id, value), as `unit_values(..., with_ids=True)`; sides as `pair_sides`)."""
    units_ids = None
    if sides is not None:
        units_ids = units
        units = {rk: {k: [v for _, v in lst] for k, lst in u.items()} for rk, u in units_ids.items()}
    populated = {rk[0] for rk, u in units.items() if u}
    oct_ = units.get(("other_cell_type", ""), {})
    rows = []
    for rung, pc, layer, holds in LADDER:
        u = units.get((rung, pc), {})
        if sides is not None and rung in TWO_SIDED and u:
            rows.append(_two_sided_row(rung, pc, layer, holds, units_ids, sides, populated, n_boot, seed))
            continue
        n_all = sum(len(v) for v in u.values())
        c_all = len({c for c, _ in u})
        low = _lower(rung, populated)
        row = dict(rung=rung, pair_class=pc, layer=layer, holds=holds, n_pairs_all=n_all, n_cells_all=c_all,
                   increment_over=low[0] if low else "", n_pairs=0, n_cells=0, n_assays=0,
                   variance=float("nan"), layer_increment=float("nan"), ci_low=float("nan"),
                   ci_high=float("nan"), ci_width=float("nan"), share=float("nan"), share_units=0)
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
        # share: on the (cell, assay) units of this increment that also hold other_cell_type pairs, the
        # increment over the other_cell_type pair value of those same units (cell means, then mean)
        su = [k for k in matched if k in oct_]
        if su:
            sc = sorted({c for c, _ in su})
            num = np.mean([np.mean([np.mean(u[k]) - (np.mean(lu[k]) if low else 0.0) for k in su if k[0] == c])
                           for c in sc])
            den = np.mean([np.mean([np.mean(oct_[k]) for k in su if k[0] == c]) for c in sc])
            row.update(share=float(num / den), share_units=len(su))
        flag = "ok" if est >= 0 else "ok; negative increment (not clipped)"
        row.update(variance=float(np.mean([var_c[c] for c in cells])), layer_increment=est, ci_low=lo,
                   ci_high=hi, ci_width=hi - lo, flag=flag, _inc_c=inc_c)
        rows.append(row)
    return rows


def _two_sided_row(rung, pc, layer, holds, units_ids, sides, populated, n_boot, seed):
    """One two-sided ladder row (see the module doc). Same cell averaging, estimability rule, bootstrap
    and interval as the one-sided row; the subtraction is pair by pair, 0.5 x (L(a) + L(b))."""
    u_ids = units_ids.get((rung, pc), {})
    low = _lower(rung, populated)
    low_by_exp = _lower_by_experiment(units_ids.get(low, {}), sides)
    n_all = sum(len(v) for v in u_ids.values())
    c_all = len({c for c, _ in u_ids})
    row = dict(rung=rung, pair_class=pc, layer=layer, holds=holds, n_pairs_all=n_all, n_cells_all=c_all,
               increment_over=low[0], n_pairs=0, n_cells=0, n_assays=0, variance=float("nan"),
               layer_increment=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
               ci_width=float("nan"), share=float("nan"), share_units=0)
    kept, n_a, n_b = _two_sided_kept(u_ids, low_by_exp, sides)
    matched = sorted(kept)
    n_m = sum(len(kept[k]) for k in matched)
    cells = sorted({c for c, _ in matched})
    row.update(n_pairs=n_m, n_cells=len(cells), n_assays=len({a for _, a in matched}), n_pairs_left_b=n_b)
    if n_m < MIN_PAIRS or len(cells) < MIN_CELLS:
        row["flag"] = (f"not estimable: {n_m} pairs from {len(cells)} cells with a {low[0]} pair in the "
                       f"experiments of both sides (rule: {MIN_PAIRS} pairs from {MIN_CELLS} cells); left: "
                       f"{n_b} pairs whose b-side experiment has no {low[0]} pair, {n_a} whose a-side has "
                       f"none; all pairs of the row: {n_all} from {c_all} cells")
        return row

    def cell_mean(ks, j):
        return float(np.mean([np.mean([x[j] for x in kept[k]]) for k in ks]))
    inc_c = {c: cell_mean([k for k in matched if k[0] == c], 1) for c in cells}
    var_c = {c: cell_mean([k for k in matched if k[0] == c], 0) for c in cells}
    est = float(np.mean([inc_c[c] for c in cells]))
    boot = _joint_boot(kept, matched, units_ids, low, sides, n_boot, seed)
    lo, hi = LY._interval(est, boot, len(cells))
    # share: on the units that also hold kept other_cell_type pairs, this increment over the kept
    # other_cell_type pair value of those units (cell means, then mean)
    oct_ids = units_ids.get(("other_cell_type", ""), {})
    oct_kept, _, _ = _two_sided_kept(oct_ids, _lower_by_experiment(units_ids.get(("isogenic", ""), {}),
                                                                   sides), sides)
    su = [k for k in matched if k in oct_kept]
    if su:
        sc = sorted({c for c, _ in su})
        num = np.mean([np.mean([np.mean([x[1] for x in kept[k]]) for k in su if k[0] == c]) for c in sc])
        den = np.mean([np.mean([np.mean([x[0] for x in oct_kept[k]]) for k in su if k[0] == c]) for c in sc])
        row.update(share=float(num / den), share_units=len(su))
    flag = "ok" if est >= 0 else "ok; negative increment (not clipped)"
    row.update(variance=float(np.mean([var_c[c] for c in cells])), layer_increment=est, ci_low=lo,
               ci_high=hi, ci_width=hi - lo, flag=flag, _inc_c=inc_c)
    return row


def _joint_boot(kept, matched, units_ids, low, sides, n_boot, seed):
    """Bootstrap draws of a two-sided increment, resampling every cell of the block jointly.

    Each draw takes |U| cells with replacement from U (every a-side cell of any rung in the block, and
    every b-side cell of the row's kept pairs); w_c is the number of times cell c is drawn, and it applies
    wherever c appears. Inside a draw: L*(e, assay) is the w-weighted mean of the lower-rung pair values
    of experiment e (each pair weighted by the w of its cell); a kept pair enters with weight w(b-side
    cell) when both L*(a) and L*(b) exist, with increment v - 0.5 (L*(a) + L*(b)); unit means are
    pair-weighted means, cell means are unweighted means over the cell's valid units, and the draw is the
    w-weighted mean of the cell means over a-side cells. With every w = 1 a draw equals the estimate.
    Returns an array of n_boot draws (nan where a draw holds no valid a-side cell)."""
    lower = [(sides[pid][0], a, c, v) for (c, a), lst in units_ids.get(low, {}).items() for pid, v in lst]
    pairs = [(k, x[0], x[2]) for k in matched for x in kept[k]]
    acells = {c for u in units_ids.values() for (c, _) in u}
    U = sorted(acells | {sides[pid][3] for _, _, pid in pairs})
    ci = {c: i for i, c in enumerate(U)}
    keys = sorted({(e, a) for e, a, _, _ in lower})
    ki = {k: i for i, k in enumerate(keys)}
    lc = np.array([ci[c] for _, _, c, _ in lower])
    lv = np.array([v for _, _, _, v in lower])
    K = np.zeros((len(lower), len(keys)))
    K[np.arange(len(lower)), [ki[(e, a)] for e, a, _, _ in lower]] = 1.0
    units = sorted({k for k, _, _ in pairs})
    ui = {u: i for i, u in enumerate(units)}
    cells = sorted({c for c, _ in units})
    cj = {c: i for i, c in enumerate(cells)}
    pv = np.array([v for _, v, _ in pairs])
    ka = np.array([ki[(sides[pid][0], k[1])] for k, _, pid in pairs])
    kb = np.array([ki[(sides[pid][1], k[1])] for k, _, pid in pairs])
    pb = np.array([ci[sides[pid][3]] for _, _, pid in pairs])
    P = np.zeros((len(pairs), len(units)))
    P[np.arange(len(pairs)), [ui[k] for k, _, _ in pairs]] = 1.0
    C = np.zeros((len(units), len(cells)))
    C[np.arange(len(units)), [cj[u[0]] for u in units]] = 1.0
    acol = np.array([ci[c] for c in cells])
    rng = np.random.default_rng(seed)
    out = np.full(n_boot, np.nan)
    for i0 in range(0, n_boot, 100):
        dr = rng.integers(0, len(U), size=(min(100, n_boot - i0), len(U)))
        W = np.stack([np.bincount(d, minlength=len(U)) for d in dr]).astype(float)
        wl = W[:, lc]
        with np.errstate(invalid="ignore", divide="ignore"):
            L = ((wl * lv) @ K) / (wl @ K)                       # draws x keys (nan: no lower pair drawn)
            inc = pv - 0.5 * (L[:, ka] + L[:, kb])               # draws x pairs
            wp = W[:, pb] * np.isfinite(inc)
            um = (np.nan_to_num(inc) * wp) @ P / (wp @ P)        # draws x units
            uv = np.isfinite(um)
            cm = (np.nan_to_num(um) @ C) / (uv.astype(float) @ C)  # draws x cells
            wa = W[:, acol] * np.isfinite(cm)
            out[i0:i0 + len(dr)] = (np.nan_to_num(cm) * wa).sum(1) / wa.sum(1)
    return out


COMMON_RUNGS = ("pseudoreplicate", "isogenic", "other_cell_type")
COMMON_COLUMNS = ["assay_class", "stratum", "band", "rung", "n_common_cells", "common_increment",
                  "common_ci_low", "common_ci_high", "row_increment", "row_ci_low", "row_ci_high", "row_n_cells",
                  "inside_row_interval", "ok"]


def common_cell_check(rows, n_boot=1000, seed=0, rungs=COMMON_RUNGS):
    """The acceptance check that replaces telescoping (PI decision of 2026-11-17, row 5). Per (assay
    class, stratum, band): the a-side cells that every rung of `rungs` holds as an estimable row; each
    rung's increment recomputed on those cells (the same per-cell increments as its row, so the same
    form); ok when there are at least MIN_CELLS common cells and every common-cell increment lies inside
    its full row's 95% interval. A group where a rung is not estimable is left out of the output."""
    out = []
    groups = {}
    for r in rows:
        groups.setdefault((r["assay_class"], r["stratum"], r["band"]), {})[(r["rung"], r["pair_class"])] = r
    for (cls, s, b), g in sorted(groups.items()):
        rr = [g.get((q, "")) for q in rungs]
        if any(r is None or "_inc_c" not in r for r in rr):
            continue
        common = sorted(set(rr[0]["_inc_c"]).intersection(*[set(r["_inc_c"]) for r in rr[1:]]))
        recs = []
        for q, r in zip(rungs, rr):
            rec = dict(assay_class=cls, stratum=s, band=b, rung=q, n_common_cells=len(common),
                       row_increment=r["layer_increment"], row_ci_low=r["ci_low"], row_ci_high=r["ci_high"],
                       row_n_cells=r["n_cells"], common_increment=float("nan"), common_ci_low=float("nan"),
                       common_ci_high=float("nan"), inside_row_interval="")
            if common:
                vals = np.array([r["_inc_c"][c] for c in common])
                est = float(vals.mean())
                rng = np.random.default_rng(seed)
                boot = vals[rng.integers(0, len(common), size=(n_boot, len(common)))].mean(1)
                lo, hi = LY._interval(est, boot, len(common))
                rec.update(common_increment=est, common_ci_low=lo, common_ci_high=hi,
                           inside_row_interval=int(r["ci_low"] <= est <= r["ci_high"]))
            recs.append(rec)
        ok = bool(len(common) >= MIN_CELLS and all(x["inside_row_interval"] == 1 for x in recs))
        out.extend(dict(x, ok=int(ok)) for x in recs)
    return out


CHAIN = (("pseudoreplicate", ""), ("isogenic", ""), ("other_cell_type", ""))
CHAIN_COLUMNS = ["assay_class", "stratum", "band", "n_units", "n_cells", "sampling", "replicate_increment",
                 "cell_type_increment", "sum_of_increments", "other_cell_type_value", "abs_diff", "ok"]


def chain_check(pair_rows, tol=1e-9):
    """The ladder's own telescoping check: on the (cell, assay) units that hold every rung of the chain
    pseudoreplicate -> isogenic -> other_cell_type, the assay-matched sampling value plus the two
    increments (cell means, then mean) against the other_cell_type value of the same units."""
    out = []
    for (cls, s, b), units in sorted(unit_values(pair_rows).items()):
        if cls == "control":
            continue
        us = [units.get(k, {}) for k in CHAIN]
        common = sorted(set(us[0]).intersection(*us[1:]))
        cells = sorted({c for c, _ in common})
        row = dict(assay_class=cls, stratum=s, band=b, n_units=len(common), n_cells=len(cells))
        if not common:
            out.append(dict(row, ok=""))
            continue
        def cm(f):
            return float(np.mean([np.mean([f(k) for k in common if k[0] == c]) for c in cells]))
        m = [lambda k, j=j: float(np.mean(us[j][k])) for j in range(3)]
        samp = cm(m[0])
        rep_ = cm(lambda k: m[1](k) - m[0](k))
        ct = cm(lambda k: m[2](k) - m[1](k))
        top = cm(m[2])
        tot = samp + rep_ + ct
        out.append(dict(row, sampling=samp, replicate_increment=rep_, cell_type_increment=ct,
                        sum_of_increments=tot, other_cell_type_value=top, abs_diff=abs(tot - top),
                        ok=bool(abs(tot - top) <= tol * (1 + abs(top)))))
    return out


def ladder_rows(pair_rows, n_boot=1000, seed=0, form="two_sided", sides=None):
    """All ladder rows. form two_sided needs sides (`pair_sides`) for every pair of pair_rows."""
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}: {form}")
    if form == "two_sided":
        if sides is None:
            raise ValueError("the two-sided form needs the experiment of each side (sides; CLI --tables)")
        miss = sorted({r["pair_id"] for r in pair_rows} - set(sides))
        if miss:
            raise ValueError(f"{len(miss)} pairs have no sides in the tables, e.g. {miss[:3]}")
    out = []
    for (cls, s, b), units in sorted(unit_values(pair_rows, with_ids=form == "two_sided").items()):
        if cls == "control":
            continue
        for r in ladder_block(units, n_boot, seed, sides if form == "two_sided" else None):
            out.append(dict(assay_class=cls, stratum=s, band=b, **r))
    return out


def _g(x, nd=3):
    return "" if not (isinstance(x, float) and math.isfinite(x)) else f"{x:.{nd}g}"


NOT_IDENTIFIED = "not identified inside cells"


def quote_rule(mixed, design, min_cells=MIN_CELLS):
    """Mixed-model rows with the E4 quote rule applied from design.tsv: a fitted factor other than cell
    that varies inside fewer than min_cells cells gets an empty fraction and interval (the same rule as
    `mixed_model.quote_rule`, for tables written without --min-contrast-cells). The design row is the one
    with the table's second-experiments mode (its residual note says "second experiments included")."""
    se = {(m["assay"], m["band"]): ("1" if "second experiments included" in m.get("note", "") else "0")
          for m in mixed if m["factor"] == "residual"}
    cw = {(d["assay"], d["second_experiments"], d["factor"]): d["cells_with_contrast"] for d in design}
    out = []
    for m in mixed:
        m = dict(m)
        k = cw.get((m["assay"], se.get((m["assay"], m["band"]), "0"), m["factor"]), "")
        if m["factor"] not in ("cell", "residual") and m["estimable"] == "1" and k != "" and \
                int(k) < min_cells and m["variance_fraction"] != "":
            m.update(variance_fraction="", ci_low="", ci_high="",
                     note=f"{NOT_IDENTIFIED}: varies inside {k} cells < {min_cells} (fitted, not quoted)")
        out.append(m)
    return out


def markdown(rows, tele=None, mixed=None, design=None, chain=None, form="one_sided", common=None, tables=None):
    lines = ["# Replicate ladder (close-out item E4)", "",
             "Stratum all, band all. Increments are assay-matched pair values (0.5 E[(Y - Y')^2] of log1p "
             "counts at each assay's common depth); intervals are 95% cluster-bootstrap intervals over cells. "
             "No row is pooled across assay classes. Full table: ladder.tsv.", ""]
    if form == "two_sided":
        lines += ["Form: two-sided (PI decision of 2026-11-17, row 5). The other_cell_type and other_experiment "
                  "rows subtract, pair by pair, 0.5 x (the isogenic pair value of the a-side experiment + that "
                  "of the b-side experiment). A pair whose b-side experiment has no isogenic pair leaves the "
                  "matched set (ladder.tsv column n_pairs_left_b). The bootstrap resamples all cells jointly: a "
                  "drawn cell counts as a-side and as b-side, and the isogenic values are recomputed in each draw." + (f" Sides from: {tables}." if tables else ""), ""]
    else:
        lines += ["Form: one-sided (every rung subtracts the lower rung of the a-side unit only). Kept for "
                  "comparison; for the other_cell_type and other_experiment rows it is not the ladder "
                  "definition since 2026-11-17.", ""]
    for cls in sorted({r["assay_class"] for r in rows}):
        lines += [f"## {cls}", "", "| rung | layer | increment [95% interval] | pairs (cells) matched | "
                  "pairs (cells) all | share of the cell-type pair value (same cells and assays) | flag |", "|---|---|---|---|---|---|---|"]
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
        lines += ["## Old telescoping check of layers.tsv (known to fail)", "",
                  f"{ok} of {len(tele)} groups telescope exactly (total_variance.telescoping_check on the "
                  "pooled rows of layers.tsv, which the ladder does not use). Known cause of the failures: the "
                  "rungs hold different cells, so each increment uses the cells it shares with the rung below "
                  "and the sum does not telescope (telescoping.tsv).", ""]
    if chain is not None:
        cc = [c for c in chain if c["stratum"] == "all" and c["band"] == "all"]
        lines += ["## Telescoping check of the ladder (chain_check.tsv)", "",
                  f"On the (cell, assay) units that hold pseudoreplicate, isogenic and other_cell_type pairs: "
                  f"{sum(1 for c in chain if c['ok'] is True)} of {sum(1 for c in chain if c['ok'] != '')} "
                  "(class, stratum, band) groups telescope exactly.", "",
                  "| assay class | units | cells | sampling | replicate | cell type | sum | other_cell_type | ok |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for c in cc:
            lines.append(f"| {c['assay_class']} | {c['n_units']} | {c['n_cells']} | {_g(c.get('sampling'), 4)} | "
                         f"{_g(c.get('replicate_increment'), 4)} | {_g(c.get('cell_type_increment'), 4)} | "
                         f"{_g(c.get('sum_of_increments'), 4)} | {_g(c.get('other_cell_type_value'), 4)} | "
                         f"{c['ok']} |")
        lines.append("")
    if common is not None:
        cg = {(c["assay_class"], c["stratum"], c["band"]): c["ok"] for c in common}
        lines += ["## Common-cell check (common_cells.tsv; replaces the telescoping check)", "",
                  f"Pseudoreplicate, isogenic and other_cell_type recomputed on the a-side cells that all three "
                  f"rows hold. Pass: at least {MIN_CELLS} common cells and each common-cell increment inside "
                  f"its row's interval. {sum(1 for v in cg.values() if v == 1)} of {len(cg)} (class, stratum, "
                  "band) groups pass.", "",
                  "| assay class | rung | common cells | increment on common cells [95% interval] | "
                  "row increment [95% interval] (cells) | inside | ok |", "|---|---|---|---|---|---|---|"]
        for c in common:
            if c["stratum"] != "all" or c["band"] != "all":
                continue
            lines.append(f"| {c['assay_class']} | {c['rung']} | {c['n_common_cells']} | "
                         f"{_g(c['common_increment'], 4)} [{_g(c['common_ci_low'], 3)}, {_g(c['common_ci_high'], 3)}] | "
                         f"{_g(c['row_increment'], 4)} [{_g(c['row_ci_low'], 3)}, {_g(c['row_ci_high'], 3)}] "
                         f"({c['row_n_cells']}) | {c['inside_row_interval']} | {c['ok']} |")
        lines.append("")
    if mixed:
        if design is None and not any(NOT_IDENTIFIED in m.get("note", "") for m in mixed):
            lines += ["WARNING: no design.tsv given and the mixed-model table carries no quote rule: "
                      "fractions of factors that vary inside fewer than 3 cells are shown.", ""]
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
    two = args.form == "two_sided"
    sides, tables = None, None
    if two:
        if not args.tables:
            raise SystemExit("ladder report --form two_sided needs --tables <dir> (pairs.tsv, measurements.tsv) "
                             "for the experiment of each side; --form one_sided runs without it")
        t = Path(args.tables)
        sides = pair_sides(_read_tsv(t / "pairs.tsv"), _read_tsv(t / "measurements.tsv"))
        tables = (f"{t} (pairs.tsv sha256 {_sha256(t / 'pairs.tsv')[:12]}, measurements.tsv sha256 "
                  f"{_sha256(t / 'measurements.tsv')[:12]})")
    rows = ladder_rows(pair_rows, args.n_boot, args.seed, args.form, sides)
    out = Path(args.out_dir)
    _write_tsv(rows, REPORT_COLUMNS + (["n_pairs_left_b"] if two else []), out / "ladder.tsv")
    common = common_cell_check(rows, args.n_boot, args.seed)
    _write_tsv(common, COMMON_COLUMNS, out / "common_cells.tsv")
    tele, chain = None, None
    if not two:   # telescoping holds only for the one-sided form; kept for comparison
        if args.layers:
            from onion_eval import total_variance as TV
            tele = TV.telescoping_check(args.layers)
            _write_tsv(tele, ["group", "innermost", "outermost", "sum_of_increments", "outermost_variance",
                              "abs_diff", "chain_ok", "ok"], out / "telescoping.tsv")
        chain = chain_check(pair_rows)
        _write_tsv(chain, CHAIN_COLUMNS, out / "chain_check.tsv")
    elif args.layers:
        print("ladder report: --layers ignored in the two-sided form (the common-cell check replaces telescoping)")
    mixed = [r for p in (args.mixed or []) for r in _read_tsv(p)]
    design = _read_tsv(args.design) if args.design else None
    if design is not None:
        mixed = quote_rule(mixed, design)
    (out / "ladder.md").write_text(markdown(rows, tele, mixed, design, chain, args.form, common, tables))
    print(f"ladder report ({args.form}): {len(rows)} rows -> {out / 'ladder.tsv'}, {out / 'ladder.md'}")
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
    r.add_argument("--form", choices=FORMS, default="two_sided",
                   help="two_sided (default, the ladder definition) or one_sided (the earlier form, comparison)")
    r.add_argument("--tables", default=None, help="dir with pairs.tsv and measurements.tsv (two_sided)")
    r.add_argument("--layers", default=None)
    r.add_argument("--mixed", nargs="*", default=None)
    r.add_argument("--design", default=None)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    return {"pairlist": cmd_pairlist, "design": cmd_design, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
