"""Numbers for COMPARE.md: the two-sided and the one-sided replicate ladder side by side.

Run from cruxvault/ with PYTHONPATH=<worktree>/src and the candii python:
    python results/closeout_E4/ladder_two_sided_2026-11-18/make_compare.py <worktree>/tools
Prints markdown tables to stdout. Reads only files in this folder and ../ladder_2026-11-18/.
"""
import csv, gzip, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1])
from onion_eval import ladder as LD  # noqa: E402

H = Path("results/closeout_E4/ladder_two_sided_2026-11-18")
E4 = Path("results/closeout_E4/ladder_2026-11-18")


def rows(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def g(x, nd=3):
    try:
        x = float(x)
    except ValueError:
        return ""
    return "" if not np.isfinite(x) else f"{x:.{nd}g}"


one = {(r["assay_class"], r["stratum"], r["band"], r["rung"], r["pair_class"]): r for r in rows(H / "one_sided/ladder.tsv")}
two = {(r["assay_class"], r["stratum"], r["band"], r["rung"], r["pair_class"]): r for r in rows(H / "two_sided/ladder.tsv")}

print("## Table 1. Stratum all, band all: every rung with a pair, both forms\n")
print("| assay class | rung | one-sided increment [95% interval] | pairs (cells) | two-sided increment [95% interval] | "
      "pairs (cells) | pairs left: b-side has no isogenic pair | relative change |")
print("|---|---|---|---|---|---|---|---|")
for k in sorted(one):
    cls, s, b, rung, pc = k
    if s != "all" or b != "all" or one[k]["n_pairs_all"] == "0":
        continue
    a, t = one[k], two[k]
    va, vt = a["layer_increment"], t["layer_increment"]
    rel = f"{100 * (float(vt) - float(va)) / abs(float(va)):+.1f}%" if va and vt else ""
    name = rung + (f" ({pc})" if pc else "")
    fa = f"{g(va, 4)} [{g(a['ci_low'])}, {g(a['ci_high'])}]" if va else "not estimable"
    ft = f"{g(vt, 4)} [{g(t['ci_low'])}, {g(t['ci_high'])}]" if vt else "not estimable"
    print(f"| {cls} | {name} | {fa} | {a['n_pairs']} ({a['n_cells']}) | {ft} | {t['n_pairs']} ({t['n_cells']}) | "
          f"{t.get('n_pairs_left_b', '')} | {rel} |")

# the one-sided subtraction on the two-sided matched pairs: separates the change of form from the change of pairs
pr = rows(E4 / "pair_values.tsv.gz")
tab = H / "tables_eic_v1_2x"
sides = LD.pair_sides(rows(tab / "pairs.tsv"), rows(tab / "measurements.tsv"))
uv = LD.unit_values([r for r in pr if r["stratum"] == "all" and r["band"] == "all"], with_ids=True)
print("\n## Table 2. Cell-type rung, stratum all, band all: the one-sided subtraction on the pairs the two-sided form keeps\n")
print("| assay class | pairs (cells) | one-sided, all its matched pairs | one-sided, two-sided pairs | two-sided | "
      "mean 0.5 (L(b) - L(a)) over the kept pairs |")
print("|---|---|---|---|---|---|")
for (cls, s, b), units in sorted(uv.items()):
    if cls == "control":
        continue
    low = LD._lower_by_experiment(units.get(("isogenic", ""), {}), sides)
    kept, _, _ = LD._two_sided_kept(units.get(("other_cell_type", ""), {}), low, sides)
    if not kept:
        continue
    cells = sorted({c for c, _ in kept})

    # one-sided on the kept pairs: pair value minus L(a), the a-side unit's isogenic mean
    la = {k: low[(sides[pid][0], k[1])] for k, v in units[("other_cell_type", "")].items() for pid, _ in v
          if (sides[pid][0], k[1]) in low}
    one_k = float(np.mean([np.mean([np.mean([x[0] - la[k] for x in kept[k]]) for k in kept if k[0] == c])
                                     for c in cells]))
    diff = float(np.mean([np.mean([np.mean([(x[0] - x[1]) - la[k] for x in kept[k]]) for k in kept if k[0] == c])
                          for c in cells]))
    a = one[(cls, s, b, "other_cell_type", "")]
    t = two[(cls, s, b, "other_cell_type", "")]
    print(f"| {cls} | {t['n_pairs']} ({t['n_cells']}) | {g(a['layer_increment'], 4)} | {g(one_k, 4)} | "
          f"{g(t['layer_increment'], 4)} | {g(diff, 3)} |")

print("\n## Table 3. Cell-type rung over all strata and signal bands: relative change two-sided vs one-sided\n")
print("| assay class | groups with both values | median relative change | min | max | groups where two-sided < one-sided |")
print("|---|---|---|---|---|---|")
for cls in sorted({k[0] for k in one}):
    ch = []
    for k in one:
        if k[0] == cls and k[3] == "other_cell_type" and one[k]["layer_increment"] and two[k]["layer_increment"]:
            va, vt = float(one[k]["layer_increment"]), float(two[k]["layer_increment"])
            if va != 0:
                ch.append((vt - va) / abs(va))
    if ch:
        ch = np.array(ch)
        print(f"| {cls} | {len(ch)} | {100 * np.median(ch):+.1f}% | {100 * ch.min():+.1f}% | {100 * ch.max():+.1f}% | "
              f"{int((ch < 0).sum())} |")

cc = rows(H / "two_sided/common_cells.tsv")
grp = {}
for c in cc:
    grp.setdefault((c["assay_class"], c["stratum"], c["band"]), []).append(c)
fail = {k: v for k, v in grp.items() if v[0]["ok"] != "1"}
print(f"\n## Table 4. Common-cell check, two-sided: {len(grp) - len(fail)} of {len(grp)} groups pass; the failing groups\n")
print("| assay class | stratum | band | common cells | rungs outside their row interval |")
print("|---|---|---|---|---|")
for k, v in sorted(fail.items()):
    out = [c["rung"] for c in v if c["inside_row_interval"] == "0"]
    print(f"| {k[0]} | {k[1]} | {k[2]} | {v[0]['n_common_cells']} | {', '.join(out) or '(fewer than 3 common cells)'} |")
