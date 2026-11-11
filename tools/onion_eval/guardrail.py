"""E4 -- the copy-leak guardrail, a hard gate of the selection rule (D13): an arm that fails is out.

The PI's definition (`miniCANDI architecture ablations.md`, copy-leak-guardrail) has two parts:
(1) perturb a masked bin's input and check that its output does not change; (2) score denoising
only on masked or thinned bins. Part 1 is the gate and is computed here. Part 2 is a scoring rule
that the scorer (E6) follows: same-view reconstruction (y_kind 3) is scored on hidden bins only
(`loss_mask` of interface C), and denoising and correction are scored against an independent draw
(the other pseudoreplicate half or the complement of a thinned input).

Part 1, the wiring check (the gate). Two perturbations of every batch, run one at a time:
  (a) the counts under `x_bin_mask` (hidden bins of available tracks) get a random positive offset;
  (b) the counts of every track with `x_avail` False get random values on every bin.
No output may move: for each, the max over batches and over "n" and "mu" of
max|out' - out| / max|out| is reported (`hidden_input_max_abs_change`,
`unavailable_input_max_abs_change`) and must be at most REL_TOL = 1e-6. With no hidden bin in the
batches the check did not run, and part 1 fails closed. The scorer runs this on several mask seeds
and passes an arm only when every mask seed passes.

The same-view copy gain (a diagnostic of ours, not part of the PI's definition; it does not gate
unless GATE_ON_COPY_GAIN is True). It measures how much a model uses a visible bin's own value
(J-invariance, Noise2Self, Batson & Royer 2019):
  same_view_copy_gain = CRPS(hidden bins) - CRPS(visible bins), on y_kind 3 entries with x_avail,
in NB CRPS units of counts; `same_view_copy_gain_relative` = gain / CRPS(hidden), and
`same_view_copy_gain_flag` = relative gain > GAIN_THRESHOLD. A J-invariant model scores both sets
alike (gain ~ 0); a model that echoes visible input scores visible bins near 0. GAIN_THRESHOLD =
0.10 was calibrated in tests/test_onion_eval_guardrail.py (`python tests/test_onion_eval_guardrail.py`
prints it), NB size 8, 15% of bins hidden, 5 seeds: the honest J-invariant model (mean of visible
neighbours) gave a relative gain of -0.006 to 0.012; models that respect the mask but blend weight w
of a visible bin's own value gave 0.16-0.18 (w 0.25), 0.29-0.31 (w 0.5) and 0.39-0.43 (w 1). On the
pilot's real data (part 1 at exactly 0.0) the relative gain was 0.57-0.71 for the U-Net rung and
0.14-0.20 for the conv-attention rung, with a mask-seed spread of 0.06-0.10: a mask-trained model
uses a visible bin's own value by design, so a high gain is expected behaviour, not a wiring leak.
With no y_kind 3 entry the gain is NaN and the flag is False.

`copy_leak_pass` = part 1 passed. GATE_ON_COPY_GAIN = True restores the old gate: `copy_leak_pass`
then also needs a finite relative gain at most GAIN_THRESHOLD (and select_rule then also needs the
gain's mask-seed spread to be small, `same_view_copy_gain_spread_ok`). Every result records the
setting under `gate_on_copy_gain`.

    python tools/onion_eval/guardrail.py --run <run dir> --tables <tables dir> [--split val]
        [--max-batches 8] [--seed 0] [--out copy_leak.json]
    python tools/onion_eval/guardrail.py --run <run dir> --tables <tables dir> --out-dir <run>/guardrail
        --val-seeds 0 --train-seeds 0:5 [--max-batches 32]
        (one job per run, one model load: <out-dir>/{val,train}_s<k>.json and summary.json, the
        part-1 verdict over every seed and the copy gain's spread per split; exit 0 iff part 1
        passes everywhere. It replaces the unversioned P2a runner `ONION/logs/p2a_guard_runner.py`.)
    One kit command per run, after training, with the setup retry and the exclude list:
        bash tools/onion_kit/remote_run.sh --after <train job id> <kit> 4 16000 01:00:00 \
            tools/onion_eval/guardrail.py --run <run> --tables <tables> --out-dir <run>/guardrail \
            --val-seeds 0 --train-seeds 0:5 --max-batches 32
Val-split windows: a seeded random sample of the evalset items per seed (`batches_for`), so the
spread across seeds includes window sampling, not the bin masks alone (DEFECTS item 9).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

REL_TOL = 1e-6
GAIN_THRESHOLD = 0.10
#: False: the gate is part 1 only (the PI's definition). True: the same-view copy gain also gates.
GATE_ON_COPY_GAIN = False
KIND_SAME_VIEW = 3
KEYS = ("hidden_input_max_abs_change", "unavailable_input_max_abs_change",
        "same_view_copy_gain", "same_view_copy_gain_flag", "copy_leak_pass")


def _run(model: Callable, batch: dict) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        out = model(batch)
    return {k: out[k].detach().cpu().double().numpy() for k in ("n", "mu")}


def _rel_change(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    worst = 0.0
    for k in ("n", "mu"):
        scale = max(float(np.nanmax(np.abs(a[k]))), 1e-30)
        worst = max(worst, float(np.nanmax(np.abs(b[k] - a[k]))) / scale)
    return worst


def _perturbed(batch: dict, where: torch.Tensor, gen: torch.Generator) -> dict:
    x = batch["x_counts"]
    noise = torch.randint(1, 50, x.shape, generator=gen, dtype=x.dtype)
    out = dict(batch)
    out["x_counts"] = torch.where(where, x + noise, x)
    return out


def copy_leak(model: Callable, batches: Iterable[dict], *, seed: int = 0) -> dict:
    """Part 1 (the gate) and the same-view copy gain; returns the pinned keys, part 1, the gain,
    its flag and the gate setting."""
    from candi.bench.distributional import p_from_mu
    from candi.metrics import nb_crps
    gen = torch.Generator().manual_seed(seed)
    hid_change, unav_change = 0.0, 0.0
    n_hidden, n_unavail = 0, 0
    crps = {"hidden": [], "visible": []}
    for batch in batches:
        base = _run(model, batch)
        L = batch["x_counts"].shape[1]
        hidden = batch["x_bin_mask"].bool() & batch["x_avail"].bool()[:, None, :]
        unavail = (~batch["x_avail"].bool())[:, None, :].expand(-1, L, -1)
        n_hidden += int(hidden.sum())
        n_unavail += int(unavail.sum())
        if hidden.any():
            hid_change = max(hid_change, _rel_change(base, _run(model, _perturbed(batch, hidden, gen))))
        if unavail.any():
            unav_change = max(unav_change,
                              _rel_change(base, _run(model, _perturbed(batch, unavail, gen))))
        same = ((batch["y_kind"] == KIND_SAME_VIEW) & batch["y_avail"].bool()
                & batch["x_avail"].bool())[:, None, :].expand(-1, L, -1).numpy()
        y = batch["y_counts"].numpy().astype(np.float64)
        hid = hidden.numpy()
        for name, sel in (("hidden", same & hid), ("visible", same & ~hid)):
            if sel.any():
                n, mu = base["n"][sel], base["mu"][sel]
                crps[name].append(nb_crps(n, p_from_mu(n, mu), y[sel]))
    c_hid = float(np.mean(np.concatenate(crps["hidden"]))) if crps["hidden"] else float("nan")
    c_vis = float(np.mean(np.concatenate(crps["visible"]))) if crps["visible"] else float("nan")
    gain = c_hid - c_vis
    rel_gain = gain / c_hid if c_hid > 0 else float("nan")
    part1 = bool(n_hidden > 0 and hid_change <= REL_TOL and unav_change <= REL_TOL)
    flag = bool(np.isfinite(rel_gain) and rel_gain > GAIN_THRESHOLD)
    gain_ok = bool(np.isfinite(rel_gain) and rel_gain <= GAIN_THRESHOLD)
    return {
        "hidden_input_max_abs_change": hid_change,
        "unavailable_input_max_abs_change": unav_change,
        "same_view_copy_gain": gain,
        "copy_leak_pass": part1 and (gain_ok or not GATE_ON_COPY_GAIN),
        "same_view_copy_gain_relative": rel_gain,
        "same_view_copy_gain_flag": flag,
        "same_view_crps_hidden": c_hid,
        "same_view_crps_visible": c_vis,
        "part1_wiring_pass": part1,
        "gate_on_copy_gain": bool(GATE_ON_COPY_GAIN),
        "rel_tol": REL_TOL,
        "same_view_copy_gain_threshold_relative": GAIN_THRESHOLD,
        "n_hidden_bins": n_hidden,
        "n_unavailable_bins": n_unavail,
    }


def batches_for(ds, split: str, max_batches: int, seed: int) -> list:
    """The guardrail's batches. Train split: the dataset's own seeded sampler (windows and masks
    both follow `seed`). Val split: each task's evalset items in a seeded order (rng [seed, 9]),
    round-robin over the tasks. The val iterator's first items are the same windows for every seed
    (only the masks change), which understated the mask-seed spread (DEFECTS item 9)."""
    if split == "train":
        return [bt for _, bt in zip(range(max_batches), ds)]
    from minicandi import contract as C
    rng = np.random.default_rng([int(seed), 9])
    by_task: dict = {}
    for it in ds.evalset_items():
        by_task.setdefault(it["task"], []).append(it)
    # each task's items in a seeded order, then round-robin over the tasks (as the val iterator
    # does), so every batch set holds reconstruction items, i.e. hidden bins for part 1
    lists = [[lst[i] for i in rng.permutation(len(lst))] for _, lst in sorted(by_task.items())]
    items = [lst[i] for i in range(max((len(x) for x in lists), default=0))
             for lst in lists if i < len(lst)]
    bs = int(ds.config["batch_size"])
    wins = [ds.item_window(it) for it in items[: max_batches * bs]]
    return [C.collate(wins[i:i + bs]) for i in range(0, len(wins), bs)]


def _seeds(text: str) -> list:
    """'0:5' -> [0..4]; '0,2' -> [0, 2]; '' -> []."""
    text = (text or "").strip()
    if not text:
        return []
    if ":" in text:
        a, b = (int(x) for x in text.split(":"))
        return list(range(a, b))
    return [int(x) for x in text.split(",")]


def summarize(results: dict) -> dict:
    """Part 1 over every (split, seed) and the copy gain's range and spread per split."""
    out = {"part1_wiring_pass": all(r["part1_wiring_pass"] for r in results.values()),
           "copy_leak_pass": all(r["copy_leak_pass"] for r in results.values()),
           "gate_on_copy_gain": bool(GATE_ON_COPY_GAIN), "runs": sorted(results)}
    for split in ("val", "train"):
        rel = [results[k]["same_view_copy_gain_relative"] for k in sorted(results)
               if k.startswith(split + "_")]
        fin = [float(v) for v in rel if v is not None and np.isfinite(v)]
        out[f"{split}_same_view_copy_gain_relative"] = fin
        out[f"{split}_same_view_copy_gain_relative_spread"] = (max(fin) - min(fin)
                                                               if len(fin) >= 2 else None)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--max-batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--out-dir", help="several seeds in one job (the P2a runner, F3): writes "
                    "<out-dir>/{val,train}_s<k>.json and summary.json; needs --val-seeds and/or "
                    "--train-seeds")
    ap.add_argument("--val-seeds", default="", help="e.g. 0 or 0,1 or 0:3")
    ap.add_argument("--train-seeds", default="", help="e.g. 0:5")
    a = ap.parse_args(argv)
    from onion_eval.score import set_threads
    set_threads()                                # the allocation, not the node's cores (F3)
    from minicandi.dataset import OnionDataset   # lazy: M1 owns these modules
    from minicandi.registry import load_run
    model, config = load_run(a.run)
    model.eval()
    if a.out_dir:
        jobs = [("val", k) for k in _seeds(a.val_seeds)] + [("train", k) for k in _seeds(a.train_seeds)]
        if not jobs:
            ap.error("--out-dir needs --val-seeds and/or --train-seeds")
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        results = {}
        for split, k in jobs:                    # one model load for every seed
            ds = OnionDataset(a.tables, split, config, k)
            res = copy_leak(model, batches_for(ds, split, a.max_batches, k), seed=k)
            res.update(split=split, seed=k, max_batches=a.max_batches)
            (out / f"{split}_s{k}.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
            results[f"{split}_s{k}"] = res
            print(json.dumps({"split": split, "seed": k, **{x: res[x] for x in (
                "part1_wiring_pass", "hidden_input_max_abs_change",
                "unavailable_input_max_abs_change", "same_view_copy_gain_relative")}}), flush=True)
        summ = summarize(results)
        (out / "summary.json").write_text(json.dumps(summ, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summ, sort_keys=True))
        return 0 if summ["copy_leak_pass"] else 1
    ds = OnionDataset(a.tables, a.split, config, a.seed)
    res = copy_leak(model, batches_for(ds, a.split, a.max_batches, a.seed), seed=a.seed)
    text = json.dumps(res, indent=2, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0 if res["copy_leak_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
