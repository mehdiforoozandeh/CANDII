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

    python tools/calib_bias.py --regime configs/regime.eic_val.json \
        --out /scratch/$USER/candi_kit/calib_bias --levels 2 8 32 128
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


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
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--regime", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--levels", type=int, nargs="+", default=[2, 8, 32, 128])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    from candi.eval import eval_cell_cond, quick_eval
    from candi.model import build_model
    from candi.train import DataSource, make_dataset, train

    device = torch.device(args.device)
    source = DataSource.resolve(store=args.regime)
    if not source.eval_pairs_declared():
        raise SystemExit("this regime declares no `eval_pairs` (D31); nothing to score")

    probe = make_dataset(source, "type1", train=True, batch_size=args.batch_size, seed=args.seed)
    torch.manual_seed(args.seed)
    model = build_model(num_assays=probe.num_assays, context_length=probe.context_bins,
                        resolution=probe.resolution, num_cells=probe.num_cells,
                        depth_center=probe.depth_center()).to(device)
    print(f"[bias] training {args.epochs} x {args.steps_per_epoch} steps to leave init behind",
          flush=True)
    train(model, source, device, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
          batch_size=args.batch_size, seed=args.seed, eval_every=0)
    model.eval()

    eval_ds = make_dataset(source, "type1", train=False, batch_size=args.eval_batch_size,
                           dsf_sampling="off", seed=args.seed, shuffle=False, h5_cache_ram=False,
                           cell_cond=eval_cell_cond(model))
    ceiling = int(eval_ds.eval_batches_per_pair())
    levels = [k for k in args.levels if k <= ceiling] or [1]
    n_windows = len(eval_ds._eval_indices)

    by_level, macro = {}, {}
    for k in levels:
        q = quick_eval(model, eval_ds, device, batches_per_pair=k, seed=args.seed,
                       return_records=True)
        by_level[k] = [r for r in q["records"] if r["split"] == "V_"]
        macro[k] = float(q["V_imp_crps"])
        print(f"[bias] k={k:<4} macro={macro[k]:.5f}  tracks={len(by_level[k])}", flush=True)

    ref = max(levels)
    rows = bias_table(by_level, ref)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "calib_bias.json").write_text(json.dumps(
        {"regime": args.regime, "levels": levels, "ref_level": ref, "macro": macro,
         "eval_windows": n_windows, "ceiling": ceiling, "rows": rows}, indent=2))

    lines = [
        "# Which tracks carry the coverage bias",
        "",
        f"- regime `{args.regime}`, {n_windows:,} eval windows, ceiling {ceiling} cycles",
        f"- reference coverage: **{ref} cycles = {ref * args.eval_batch_size:,} windows/target**",
        f"- **{len(rows)}** tracks present at every coverage (intersected, never unioned)",
        "",
        "## Macro, and how the per-track bias is distributed",
        "",
        "| coverage | windows/target | macro | mean bias | median bias | worst track |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in levels:
        b = np.array([r[f"bias@{k}"] for r in rows], dtype=float)
        worst = max(rows, key=lambda r: abs(r[f"bias@{k}"]))
        lines.append(
            f"| {k * args.eval_batch_size / max(1, n_windows):.2%} | "
            f"{k * args.eval_batch_size:,} | {macro[k]:.4f} | {b.mean():+.4f} | "
            f"{np.median(b):+.4f} | {worst[f'bias@{k}']:+.4f} "
            f"({worst['imp_biosample']}\\|{worst['assay']}) |")
    lines += ["", "## Per track, at the shipped coverage", "",
              "Sorted by how far that coverage puts the track from the reference.", "",
              "| target | assay | reference | shipped | bias |", "|---|---|---:|---:|---:|"]
    k0 = min(levels)
    for r in sorted(rows, key=lambda r: -abs(r[f"bias@{k0}"])):
        lines.append(f"| {r['imp_biosample']} | {r['assay']} | {r['ref']:.4f} | "
                     f"{r[f'crps@{k0}']:.4f} | {r[f'bias@{k0}']:+.4f} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[bias] wrote {out / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
