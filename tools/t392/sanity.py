"""t392 chunk C10a: a sanity check of a read-start run against the read-start truth (R12).

    python tools/t392/sanity.py --run <run dir> --evalset <evalset json> --out <json>

On the imputation targets of the evaluation set (split `--split`, default val) and on its pair
regions (`pair_regions` when the file has them, else every region), the inner model of the run
(the model under the read-start adapter, before any conversion to overlap counts) predicts
`mu_sub`, `n_sub` [B, L, A, 2]. The truth is the target's `y_counts_sub` from the read-start
store, read by the adapter's own `ReadStartDataset`. Bins are every bin of every window, on the
channels that `read_start.channel_mask()` keeps (both strands of a stranded track, channel 0 of
an unstranded one). Each window holds one target and only its track is read.

Two numbers, pooled over all those bins:
  nb_deviance_mean   the mean NB unit deviance per bin,
                     2 [y log(y / mu) - (y + n) log((y + n) / (mu + n))], with y log(y / mu) = 0 at y = 0
  pearson_log1p      Pearson r of log1p(mu) against log1p(y)

This is a sanity check only: it is not a scorer metric and no headline term comes from it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import contract as C  # noqa: E402
from minicandi.read_start import channel_mask  # noqa: E402


def nb_deviance(y: np.ndarray, mu: np.ndarray, n: np.ndarray) -> np.ndarray:
    """The NB unit deviance per element (float64)."""
    y, mu, n = (np.asarray(a, np.float64) for a in (y, mu, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    t2 = (y + n) * np.log((y + n) / (mu + n))
    return 2.0 * (t1 - t2)


def _regions(ev: dict) -> list:
    regs = ev["regions"]
    idx = ev.get("pair_regions")
    return [regs[i] for i in idx] if idx is not None else list(regs)


def run_sanity(run, evalset, split: str = "val", log=print) -> dict:
    from minicandi.registry import load_run
    from minicandi.rs_adapter import ReadStartAdapter
    model, config = load_run(run)
    if not isinstance(model, ReadStartAdapter):
        raise ValueError(f"{run}: not a read-start run (load_run did not return the adapter)")
    inner = model.inner.eval()
    ds = model.ds
    ev = json.loads(Path(evalset).read_text(encoding="utf-8"))
    rb = int(ev["region_bins"])
    L, bs = ds.L, int(config.get("batch_size", 8))
    regions = _regions(ev)
    jobs = [j for j in ev["imputation_targets"]
            if j.get("split", split) == split and j["target"] in ds.rows]
    keep = channel_mask()
    ys, mus, ns = [], [], []
    n_windows = 0
    for j in jobs:
        t = C.TRACK_INDEX[ds.rows[j["target"]]["assay"]]
        inputs = [m for m in j["inputs"] if m in ds.rows]
        starts = []
        for chrom, s0 in regions:
            n_bins = ds.genome.chrom_sizes.get(chrom, 0) // C.RES
            starts += [(chrom, int(s0) + k * L) for k in range(max(rb // L, 1))
                       if int(s0) + (k + 1) * L <= n_bins]
        for i in range(0, len(starts), bs):
            b = C.collate([ds.window(c, s, inputs, [j["target"]]) for c, s in starts[i:i + bs]])
            with torch.no_grad():
                out = inner(b)
            ch = keep[t]
            ys.append(b["y_counts_sub"][:, :, t][..., ch].numpy().ravel())
            mus.append(out["mu_sub"][:, :, t][..., ch].double().numpy().ravel())
            ns.append(out["n_sub"][:, :, t][..., ch].double().numpy().ravel())
            n_windows += len(starts[i:i + bs])
    if not ys:
        raise ValueError(f"{evalset}: no imputation target of split {split} in the run's tables")
    y, mu, n = np.concatenate(ys), np.concatenate(mus), np.concatenate(ns)
    dev = nb_deviance(y, mu, n)
    r = float(np.corrcoef(np.log1p(mu), np.log1p(y))[0, 1])
    res = {"nb_deviance_mean": float(dev.mean()), "pearson_log1p": r,
           "n_bins": int(y.size), "n_targets": len(jobs), "n_windows": n_windows,
           "n_regions": len(regions), "split": split, "run": str(run), "evalset": str(evalset),
           "note": "sanity check only (t392 R12): inner model vs read-start truth, not a scorer metric"}
    log(f"sanity: deviance/bin {res['nb_deviance_mean']:.4g}, pearson log1p {r:.4g} "
        f"over {res['n_bins']} bins, {len(jobs)} targets, {len(regions)} regions")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val")
    a = ap.parse_args(argv)
    res = run_sanity(a.run, a.evalset, a.split)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
