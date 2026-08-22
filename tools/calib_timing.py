"""Calibration (a) — what a mid-training check COSTS, and whether it can tell epochs apart.

Calibration (b) (`tools/window_content.py`) priced coverage from the data alone: on chr21 the
shipped 8 windows per target carry a relative standard error of 1.79, and half of all windows hold
no foreground bin at all. What (b) could not supply is the constant — it scored a surrogate
(predict-the-mean), not a trained model — nor the two numbers a coverage choice actually needs:

1. **What does a check cost?** Seconds, at each coverage, on the real store and the real model.
2. **Can it tell epochs apart?** A selection metric is only worth running if its epoch-to-epoch
   improvement is larger than its own jitter. Below that line, best-checkpoint selection is
   choosing noise, and no amount of it helps.

ONE training run answers both. At every eval epoch the hook scores the same model state at every
coverage level in turn, so the seconds-vs-coverage curve comes from one set of weights and the
epoch-to-epoch curve comes from one run. Training six times at six coverages would spend six times
the GPU to learn less, because the six runs would also differ from each other.

The instrument is `train()` itself, driven with a custom `eval_hook` — not a reimplementation of
the loop. A calibration measured against a copy of the training loop would be calibrating the copy.

    python tools/calib_timing.py --regime /…/regime_eic.json \
        --out /scratch/$USER/candi_kit/calib_a --epochs 10 --levels 1 2 4 8 16
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def gain_vs_jitter(curve: list) -> dict:
    """Does the metric move more between epochs than it wobbles?

    `gain` is the typical epoch-to-epoch improvement — the median of `c[i] - c[i+1]`, positive when
    the metric is falling, which for CRPS is better.

    `jitter` is the median absolute SECOND difference, `|c[i] - 2c[i+1] + c[i+2]|`. A curve that
    descends smoothly has a second difference near zero however steeply it falls, so this measures
    roughness WITHOUT being fooled by a strong trend — which a plain standard deviation would be.

    `ratio` is gain / jitter. Below 1 the step between two epochs is smaller than the wobble
    between three, so the ordering of two nearby checkpoints is not information. This is the
    number the coverage choice turns on, and it is reported per level rather than argued about.
    """
    c = [x for x in curve if np.isfinite(x)]
    if len(c) < 3:
        return {"n": len(c), "gain": float("nan"), "jitter": float("nan"),
                "ratio": float("nan")}
    d1 = [c[i] - c[i + 1] for i in range(len(c) - 1)]
    d2 = [abs(c[i] - 2 * c[i + 1] + c[i + 2]) for i in range(len(c) - 2)]
    gain, jit = float(np.median(d1)), float(np.median(d2))
    return {"n": len(c), "gain": gain, "jitter": jit,
            "ratio": (gain / jit if jit > 0 else float("inf"))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--regime", required=True, help="store regime json")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--levels", type=int, nargs="+", default=[2, 8, 32, 128],
                   help="batches_per_pair values to price, cheapest first")
    p.add_argument("--anchor-level", type=int, default=0,
                   help="a coverage run ONCE, at the final epoch only (0 = off). Pass the full "
                        "cycle count to score whole chr21 per target: every cheap level's bias is "
                        "then measurable against the number it is trying to estimate, rather than "
                        "argued about. Too expensive to run every epoch, and it does not need to "
                        "be -- bias is a property of the coverage, not of the epoch.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heads", default="count")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    from candi.eval import eval_cell_cond, quick_eval
    from candi.model import build_model
    from candi.train import DataSource, make_dataset, train

    device = torch.device(args.device)
    source = DataSource.resolve(store=args.regime)
    if not source.eval_pairs_declared():
        raise SystemExit("this regime declares no `eval_pairs`, so there is nothing to score "
                         "(D31). Calibration (a) needs imputation targets.")

    probe = make_dataset(source, "type1", train=True, batch_size=args.batch_size, seed=args.seed)
    torch.manual_seed(args.seed)
    model = build_model(num_assays=probe.num_assays, context_length=probe.context_bins,
                        resolution=probe.resolution, num_cells=probe.num_cells,
                        depth_center=probe.depth_center(), heads=args.heads).to(device)
    n_par = sum(q.numel() for q in model.parameters())

    # ONE eval dataset, opened here and re-iterated at every check and every level (t28). Every
    # number below therefore comes off the SAME window set, so the levels are nested rather than
    # merely similar: level k's windows are a subset of level 2k's, and a difference between two
    # levels is coverage and nothing else.
    eval_ds = make_dataset(source, "type1", train=False, batch_size=args.eval_batch_size,
                           dsf_sampling="off", seed=args.seed, shuffle=False, h5_cache_ram=False,
                           cell_cond=eval_cell_cond(model))
    n_windows_total = len(eval_ds._eval_indices)
    n_slots = sum(max(1, len(eval_ds._all_imp_biosamples(t))) for t in eval_ds._bios_candidates())
    n_cycles = max(1, n_windows_total // args.eval_batch_size // n_slots)
    levels = [k for k in args.levels if k <= n_cycles] or [1]
    print(f"[calib-a] {n_par:,} params, {n_windows_total:,} eval windows, {n_slots} slots, "
          f"{n_cycles} cycles; pricing batches_per_pair={levels}", flush=True)

    rows: list = []

    def eval_hook(step, ep):
        model.eval()
        last = (ep == args.epochs - 1)
        todo = list(levels) + ([args.anchor_level] if (args.anchor_level and last) else [])
        for k in todo:
            t0 = time.time()
            q = quick_eval(model, eval_ds, device, batches_per_pair=k, seed=args.seed)
            rows.append({"epoch": ep, "step": step, "level": k, "anchor": k not in levels,
                         "windows_per_target": k * args.eval_batch_size,
                         "wall_s": round(time.time() - t0, 2),
                         "V_imp_crps": float(q["V_imp_crps"]),
                         "V_n_targets": int(q["V_n_targets"]),
                         "n_units": int(q["n_units"])})
            print(f"[calib-a] ep{ep} k={k:<3} {rows[-1]['wall_s']:>7.2f}s  "
                  f"V_imp_crps={rows[-1]['V_imp_crps']:.5f}  n={rows[-1]['V_n_targets']}",
                  flush=True)
        model.train()

    t_train = time.time()
    train(model, source, device, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
          batch_size=args.batch_size, seed=args.seed, eval_hook=eval_hook, eval_every=1)
    train_s = time.time() - t_train

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_level = {}
    for k in levels:
        rs = [r for r in rows if r["level"] == k]
        per_level[k] = {
            "windows_per_target": k * args.eval_batch_size,
            "coverage": (k * args.eval_batch_size * n_slots) / max(1, n_windows_total),
            "median_wall_s": float(np.median([r["wall_s"] for r in rs])) if rs else float("nan"),
            "curve": [r["V_imp_crps"] for r in rs],
            **gain_vs_jitter([r["V_imp_crps"] for r in rs]),
        }
    (out / "calib_a.json").write_text(json.dumps(
        {"regime": args.regime, "epochs": args.epochs, "levels": levels, "params": n_par,
         "eval_windows": n_windows_total, "slots": n_slots, "cycles": n_cycles,
         "train_wall_s": round(train_s, 1), "rows": rows, "per_level": per_level}, indent=2))

    lines = [
        "# Calibration (a) — what a mid-training check costs, and whether it can tell epochs apart",
        "",
        f"- regime `{args.regime}`, {args.epochs} epochs x {args.steps_per_epoch} steps, "
        f"batch {args.batch_size}, heads `{args.heads}`",
        f"- model {n_par:,} params on `{args.device}`",
        f"- eval split: **{n_windows_total:,}** windows, {n_slots} slots, {n_cycles} cycles; "
        f"eval batch {args.eval_batch_size}",
        f"- whole run (training + every check at every level) {train_s / 60:.1f} min",
        "",
        "## The two numbers",
        "",
        "`gain` is the typical epoch-to-epoch improvement. `jitter` is the median absolute second",
        "difference — roughness that a steep but smooth descent does NOT inflate. **A ratio below 1",
        "means the step between two epochs is smaller than the wobble between three, so the",
        "ordering of two nearby checkpoints carries no information.**",
        "",
        "| batches_per_pair | windows/target | coverage | median s/check | gain | jitter | gain/jitter |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in levels:
        d = per_level[k]
        flag = "" if not np.isfinite(d["ratio"]) or d["ratio"] >= 1.0 else "  ← below 1"
        lines.append(
            f"| {k} | {d['windows_per_target']:,} | {d['coverage']:.2%} | "
            f"{d['median_wall_s']:.1f} | {d['gain']:.5f} | {d['jitter']:.5f} | "
            f"{d['ratio']:.2f}{flag} |")
    anchor_rows = [r for r in rows if r.get("anchor")]
    if anchor_rows:
        a = anchor_rows[0]
        last_ep = a["epoch"]
        lines += [
            "",
            "## Bias against whole chr21, at the final epoch",
            "",
            f"The anchor scored **{a['windows_per_target']:,} windows per target** "
            f"(batches_per_pair={a['level']}) in {a['wall_s']:.1f}s, giving "
            f"**{a['V_imp_crps']:.5f}**. Every cheap level is trying to estimate that number; the",
            "gap below is how far off it lands on this model. A level whose bias is larger than the",
            "epoch-to-epoch gain above cannot be compared to the final test number, however stable",
            "it is -- stability and accuracy are different failures.",
            "",
            "| batches_per_pair | coverage | value | bias vs whole chr21 | s/check |",
            "|---:|---:|---:|---:|---:|",
        ]
        for k in levels:
            v = [r for r in rows if r["epoch"] == last_ep and r["level"] == k]
            if not v:
                continue
            d = v[0]
            lines.append(f"| {k} | {per_level[k]['coverage']:.2%} | {d['V_imp_crps']:.5f} | "
                         f"{d['V_imp_crps'] - a['V_imp_crps']:+.5f} | {d['wall_s']:.1f} |")
    lines += ["", "## The curve, per level", "",
              "| epoch | " + " | ".join(f"k={k}" for k in levels) + " |",
              "|---:|" + "---:|" * len(levels)]
    for ep in sorted({r["epoch"] for r in rows}):
        cells = []
        for k in levels:
            v = [r["V_imp_crps"] for r in rows if r["epoch"] == ep and r["level"] == k]
            cells.append(f"{v[0]:.5f}" if v else "—")
        lines.append(f"| {ep} | " + " | ".join(cells) + " |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[calib-a] wrote {out / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
