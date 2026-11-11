"""E4 -- the copy-leak guardrail, a hard gate of the selection rule (D13): an arm that fails is out.

Part 1, the wiring check. Two perturbations of every batch, run one at a time:
  (a) the counts under `x_bin_mask` (hidden bins of available tracks) get a random positive offset;
  (b) the counts of every track with `x_avail` False get random values on every bin.
No output may move: for each, the max over batches and over "n" and "mu" of
max|out' - out| / max|out| is reported (`hidden_input_max_abs_change`,
`unavailable_input_max_abs_change`) and must be at most REL_TOL = 1e-6.

Part 2, the scoring rule and the J-invariance check (Noise2Self, Batson & Royer 2019). Same-view
reconstruction (y_kind 3) is scored on hidden bins only (`loss_mask` of interface C); this module
measures whether a model gains from seeing a bin's own value:
  same_view_copy_gain = CRPS(hidden bins) - CRPS(visible bins), on y_kind 3 entries with x_avail,
in NB CRPS units of counts. A J-invariant model scores both sets alike (gain ~ 0); a model that
echoes visible input scores visible bins near 0 (gain ~ CRPS(hidden)). Because the gain scales with
depth, the gate is relative: it fails when gain > GAIN_THRESHOLD x CRPS(hidden).
GAIN_THRESHOLD = 0.10 was calibrated in tests/test_onion_eval_guardrail.py (`python
tests/test_onion_eval_guardrail.py` prints it), NB size 8, 15% of bins hidden, 5 seeds: the honest
J-invariant model (mean of visible neighbours) gave a relative gain of -0.006 to 0.012; models that
respect the mask but blend weight w of a visible bin's own value gave 0.16-0.18 (w 0.25),
0.29-0.31 (w 0.5) and 0.39-0.43 (w 1). The identity-copy model, which also reads hidden bins,
fails part 1. With no y_kind 3 entry the gain is NaN and the gate fails,
because the check was not run.

`copy_leak_pass` = part 1 passed and part 2 passed. Denoising against an independent draw (the other
pseudoreplicate half or the complement of a thinned input) is the scorer's job (E6); this module
does not score it.

    python tools/onion_eval/guardrail.py --run <run dir> --tables <tables dir> [--split val]
        [--max-batches 8] [--seed 0] [--out copy_leak.json]
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
KIND_SAME_VIEW = 3
KEYS = ("hidden_input_max_abs_change", "unavailable_input_max_abs_change",
        "same_view_copy_gain", "copy_leak_pass")


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
    """Run both parts of the guardrail; returns the pinned keys plus the parts and the threshold."""
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
    part1 = bool(hid_change <= REL_TOL and unav_change <= REL_TOL)
    part2 = bool(np.isfinite(rel_gain) and rel_gain <= GAIN_THRESHOLD)
    return {
        "hidden_input_max_abs_change": hid_change,
        "unavailable_input_max_abs_change": unav_change,
        "same_view_copy_gain": gain,
        "copy_leak_pass": part1 and part2,
        "same_view_copy_gain_relative": rel_gain,
        "same_view_crps_hidden": c_hid,
        "same_view_crps_visible": c_vis,
        "part1_wiring_pass": part1,
        "part2_j_invariance_pass": part2,
        "rel_tol": REL_TOL,
        "gain_threshold_relative": GAIN_THRESHOLD,
        "n_hidden_bins": n_hidden,
        "n_unavailable_bins": n_unavail,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--max-batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    from minicandi.dataset import OnionDataset   # lazy: M1 owns these modules
    from minicandi.registry import load_run
    model, config = load_run(a.run)
    model.eval()
    ds = OnionDataset(a.tables, a.split, config, a.seed)
    batches = [bt for _, bt in zip(range(a.max_batches), ds)]
    res = copy_leak(model, batches, seed=a.seed)
    text = json.dumps(res, indent=2, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0 if res["copy_leak_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
