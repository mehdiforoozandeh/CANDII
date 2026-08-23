"""Calibration (a), part 2 — WHICH tracks carry the coverage bias.

Calibration (a) measured the bias in the macro mean: at the shipped 0.45% coverage the mid-training
score reads 0.152 too low, and the bias is gone by ~7%. What it could not say is whether that is a
broad property of the chromosome or a few tracks dragging the mean, because `quick_eval` collapses
its per-track rows before returning. It now returns them (`return_records=True`), and this script
scores one model state at several coverages and differences the rows.

TRAINING BARELY MATTERS HERE, and the previous run is the evidence. Across ten epochs the bias
against the densest level moved by 0.0057 at the shipped coverage and 0.0010 at 7%, while the model
itself moved 0.0215 per epoch. The bias is a property of WHICH WINDOWS a coverage samples, not of
how good the model is — so a short run measures it as well as a long one, and this script trains
just enough to leave the initialisation behind.

**RETIRED as a runner.** `main()` refuses, for the reason its docstring gives. The prose here is
the record of what calibration (a) part 2 asked; `bias_table` is the part still live and tested.
"""
from __future__ import annotations


def bias_table(by_level: dict, ref_level: int) -> list:
    """Per track: its score at each coverage, and the gap to the reference coverage.

    Keyed on `(biosample, imp_biosample, assay)` and INTERSECTED across levels, never unioned. A
    track missing from one level is dropped from the comparison entirely: a mean taken over a
    different track set at each coverage would itself be a coverage effect, which is exactly the
    thing being measured.
    """
    keyed = {lv: {(r["biosample"], r["imp_biosample"], r["assay"]): r for r in recs}
             for lv, recs in by_level.items()}
    common = set.intersection(*(set(k) for k in keyed.values())) if keyed else set()
    rows = []
    for key in sorted(common):
        row = {"biosample": key[0], "imp_biosample": key[1], "assay": key[2]}
        ref = keyed[ref_level][key]["crps"]
        row["ref"] = float(ref)
        row["n_points_ref"] = int(keyed[ref_level][key]["n_points"])
        for lv in sorted(keyed):
            row[f"crps@{lv}"] = float(keyed[lv][key]["crps"])
            row[f"bias@{lv}"] = float(keyed[lv][key]["crps"] - ref)
        rows.append(row)
    return rows


def main(argv=None) -> int:
    """RETIRED — the instrument it drove no longer exists, so it refuses rather than crashes.

    It scored one model state at several `batches_per_pair` coverages and differenced the per-track
    rows `quick_eval(return_records=True)` handed back. `candi.eval` is deleted (D15); the
    mid-training scorer is `candi.monitor`, which scores every 25 bp bin and therefore has no
    coverage to vary — and it returns its per-track rows unconditionally (`DialResult.per_track`),
    so the collapse this measurement worked around is gone too. `bias_table` above is kept: it is
    the table t31 published, and it is a pure function of rows anything can supply.
    """
    raise SystemExit(
        "tools/calib_bias.py is RETIRED. It differenced `candi.eval.quick_eval` rows across "
        "`batches_per_pair` coverages, and both are gone: `candi.eval` was deleted (D15) and "
        "`candi.monitor` scores every bin of the eval chromosomes at one fixed coverage. The t31 "
        "result stands as recorded; nothing re-runs it. `bias_table` in this file is still live "
        "and still tested.")


if __name__ == "__main__":
    raise SystemExit(main())
