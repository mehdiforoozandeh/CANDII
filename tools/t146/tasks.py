"""t146 — the final CAC testbed: the index -> run table, the pair sets and the leakage audit.

All runs are ONE shared design-D model (old rung D: 4 dilated conv layers x 32 channels, kernel 5,
dilations 1/2/4/8, FiLM from g, g hidden 64) over all 7 counterfactual-arm tracks (g = "all"),
count space, NB output, real covariates, depth 7.5M held out of training (as t126/t128), 12,000-step
cap with early stopping on chr22 (patience 4, eval every 250 steps) — old shared D's settings
(t128 phase 2, `D_all_counts_real_s<seed>`).

**Arms** (`ARMS`; h = the hypothesis it serves):
  alone     h29/h30/h31 comparator. Loss = L_corr (the correction NB NLL, unscaled, as old D).
  matched   h29 null control. Loss = L_corr / L0_corr: the correction term scaled by EXACTLY the
            factor the joint runs apply to it (L0_corr = the step-0 correction loss, measured the
            same way and from the same seed as the joint runs; `train.step0_losses`).
  xassay    h29. Loss = L_corr / L0_corr + w * L_imp / L0_imp. L_imp = NB NLL of the cross-assay
            imputation head: source = the BASE track of assay A, target = a product of assay B of
            the same cell (`xassay_pairs`), g fed (C_baseA, C_targetB).
  denoise   h29. Loss = L_corr / L0_corr + w * L_den / L0_den. L_den = NB NLL, on the masked bins
            only, of the source product's own counts from its masked window (15 % of input bins
            zeroed, a mask-indicator input channel), g fed (C_p, C_p), its own head.
  consist   h30. Loss = L_corr + W_CONSIST * L_cons. L_cons = mean over bins of
            (loc_direct - loc_chained)^2 + (disp_direct - disp_chained)^2 on a never-trained
            arm->arm pair (a, a'); chained = a -> base -> a' through the same model; gradient flows
            through both routes (the stop-gradient target collapsed in the smoke).
  map       h31. + per-bin (25 bp) Umap k36 mappability as a second input channel.
  mapshift  h31 null control. + the same channel circularly shifted by 5 Mb within the chromosome.
  dna       h31. + one-hot DNA (A, C, G, T; N = all zero) through a small conv encoder.

**Tasks** (42 rows; `tasks()`; the PI-approved design counts 24 + 9 + 9):
  0-2 alone s0-2 | 3-5 alone s3-5 | 6-8 matched s0-2 | 9-17 xassay w x s (w slowest) |
  18-26 denoise w x s | 27-32 consist s0-5 | 33-35 map s0-2 | 36-38 mapshift s0-2 | 39-41 dna s0-2

    python tools/t146/tasks.py tasks
    python tools/t146/tasks.py audit <MANIFEST.tsv> <covariates.tsv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import encoding, pairs as lp  # noqa: E402

G = "all"
SPACE = "counts"
RUNG = "D"
HOLDOUT = (("depth", "7.5M"),)
MAX_STEPS = 12000                  # old shared D (t128 phase 2 arm D)
WEIGHTS = (0.25, 1.0, 4.0)         # h29 imputation / denoising weight; 1 is primary
W_CONSIST = 1.0                    # h30 consistency weight (fixed: L_cons ~ 0 at init, no step-0 norm)
MASK_RATE = 0.15                   # h29 denoising: fraction of input bins masked
SHIFT_BINS = 200_000               # h31 null: 5 Mb / 25 bp
TASK_KEYS = ("index", "arm", "hyp", "weight", "seed", "inputs", "run_name")

ARMS = {
    "alone":    {"hyp": "h29,h30,h31", "aux": None, "corr_scale": "none", "inputs": ()},
    "matched":  {"hyp": "h29", "aux": None, "corr_scale": "step0", "inputs": ()},
    "xassay":   {"hyp": "h29", "aux": "xassay", "corr_scale": "step0", "inputs": ()},
    "denoise":  {"hyp": "h29", "aux": "denoise", "corr_scale": "step0", "inputs": ("mask",)},
    "consist":  {"hyp": "h30", "aux": "consist", "corr_scale": "none", "inputs": ()},
    "map":      {"hyp": "h31", "aux": None, "corr_scale": "none", "inputs": ("map",)},
    "mapshift": {"hyp": "h31", "aux": None, "corr_scale": "none", "inputs": ("mapshift",)},
    "dna":      {"hyp": "h31", "aux": None, "corr_scale": "none", "inputs": ("dna",)},
}
#: arms scored through the base too (h30): chained mean, chained mixture, ceiling
CHAIN_ARMS = ("alone", "consist")
#: one config per new code path, seed 0, weight 1 (indices into `tasks()`)
SMOKE_RUNS = ("alone_s0", "matched_s0", "xassay_w1_s0", "denoise_w1_s0", "consist_s0", "map_s0",
              "mapshift_s0", "dna_s0")


def run_name(arm: str, weight, seed: int) -> str:
    w = "" if weight is None else f"_w{weight:g}"
    return f"{arm}{w}_s{int(seed)}"


def tasks() -> list[dict]:
    """The 42-row table; a pure function of the constants."""
    spec = ([("alone", None, s) for s in range(6)]
            + [("matched", None, s) for s in range(3)]
            + [("xassay", w, s) for w in WEIGHTS for s in range(3)]
            + [("denoise", w, s) for w in WEIGHTS for s in range(3)]
            + [("consist", W_CONSIST, s) for s in range(6)]
            + [(a, None, s) for a in ("map", "mapshift", "dna") for s in range(3)])
    out = []
    for arm, w, s in spec:
        out.append({"index": len(out), "arm": arm, "hyp": ARMS[arm]["hyp"],
                    "weight": w, "seed": s, "inputs": ",".join(ARMS[arm]["inputs"]) or "-",
                    "run_name": run_name(arm, None if arm == "consist" else w, s)})
    return out


def task_by_name(name: str) -> dict:
    for t in tasks():
        if t["run_name"] == name:
            return t
    raise KeyError(name)


SMOKE_INDICES = tuple(task_by_name(n)["index"] for n in SMOKE_RUNS)


# ---------------------------------------------------------------------------------------------
# pair sets
# ---------------------------------------------------------------------------------------------


def _held(pid_row: dict) -> bool:
    return (pid_row["arm"], pid_row["level"]) in {(a, lv) for a, lv in HOLDOUT}


def train_pairs(rows) -> list[dict]:
    """The correction pairs: base<->arm of every track, held-out products removed (old shared D)."""
    return lp.train_pairs(rows, G, HOLDOUT)


def fit_pids(rows) -> list[str]:
    return lp.fit_pids(rows, G, HOLDOUT)


def law_pairs(rows) -> list[dict]:
    """The scored never-trained arm->arm pairs (2094 on the t112 manifest), held-out products
    included, exactly t128's law set."""
    return lp.law_pairs(rows, G)


def consist_pairs(rows) -> list[dict]:
    """h30 consistency pairs: never-trained arm->arm pairs with neither end held out."""
    return [p for p in law_pairs(rows) if not lp._held(p, HOLDOUT)]


def xassay_pairs(rows) -> list[dict]:
    """h29 imputation pairs: for every cell with two tracks (C07, C19, C40 on the t112 manifest;
    DNase C12M02 has no partner) and each ordered (track A, track B): source = A's BASE product,
    target = every usable non-held-out product of B (its base included). direction "xassay"."""
    prods = lp.usable_products(rows)
    by_cell: dict = {}
    for r in prods:
        by_cell.setdefault(r["cell"], set()).add(r["track"])
    out = []
    for cell in sorted(by_cell):
        tracks = sorted(by_cell[cell])
        for ta in tracks:
            base_a, _ = lp._by_track(rows, ta)
            for tb in tracks:
                if tb == ta:
                    continue
                for r in sorted((r for r in prods if r["track"] == tb), key=lambda r: r["pid"]):
                    if _held(r):
                        continue
                    p = lp._pair(base_a, r, "xassay")
                    p["track_tgt"] = tb
                    p["assay_tgt"] = r["assay"]
                    out.append(p)
    return out


def base_pid_of(rows, pid: str) -> str:
    by_pid = {r["pid"]: r for r in lp.usable_products(rows)}
    base, _ = lp._by_track(rows, by_pid[pid]["track"])
    return base["pid"]


# ---------------------------------------------------------------------------------------------
# leakage audit (h29 verifiable `leakaudit`)
# ---------------------------------------------------------------------------------------------


def leakage_audit(rows, enc: encoding.Encoder, trained: list[dict], scored: list[dict]) -> dict:
    """Every (source, target) pair any training objective feeds g, against the scored
    never-trained arm->arm pairs. Fails (ValueError) if a trained pair equals a scored pair by pid
    or by its encoded 40-d covariate vector, or if an imputation source is not a base product.
    `trained` = dicts with source_pid, target_pid, direction (correction, xassay, denoise, ...)."""
    by_pid = {r["pid"]: r for r in lp.usable_products(rows)}
    scored_pid = {(p["source_pid"], p["target_pid"]) for p in scored}
    scored_vec = {enc.encode_pair(s, t).tobytes(): (s, t) for s, t in scored_pid}
    pid_hits, vec_hits, bad_src = [], [], []
    for p in trained:
        key = (p["source_pid"], p["target_pid"])
        if key in scored_pid:
            pid_hits.append(key)
        v = enc.encode_pair(*key).tobytes()
        if v in scored_vec and p.get("direction") != "consist":
            vec_hits.append({"trained": key, "scored": scored_vec[v]})
        if p.get("direction") == "xassay" and by_pid[p["source_pid"]]["arm"] != "base":
            bad_src.append(key)
    dirs = sorted({p.get("direction", "?") for p in trained})
    out = {"n_trained_pairs": len(trained), "directions": dirs,
           "n_scored_pairs": len(scored_pid),
           "pid_overlap": [list(k) for k in pid_hits if True],
           "covariate_vector_overlap": vec_hits,
           "xassay_sources_not_base": [list(k) for k in bad_src],
           "xassay_sources": sorted({p["source_pid"] for p in trained
                                     if p.get("direction") == "xassay"})}
    pid_bad = [k for k in pid_hits
               if not any(p.get("direction") == "consist" and (p["source_pid"], p["target_pid"]) == k
                          for p in trained)]
    out["pass"] = not (pid_bad or vec_hits or bad_src)
    out["note"] = ("consist pairs ARE scored arm->arm covariate pairs by design (h30: g is asked "
                   "for the direct map and pulled toward the chained route; no target data of the "
                   "pair is used) and are exempt; every other overlap fails")
    if not out["pass"]:
        raise ValueError(f"leakage audit failed: pid {pid_bad[:3]}, vector {vec_hits[:3]}, "
                         f"non-base xassay sources {bad_src[:3]}")
    return out


def _tsv(records, keys) -> str:
    return "".join("\t".join("-" if r[k] is None else str(r[k]) for k in keys) + "\n"
                   for r in records)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks")
    a = sub.add_parser("audit")
    a.add_argument("manifest")
    a.add_argument("covariates")
    args = ap.parse_args(argv)
    if args.cmd == "tasks":
        sys.stdout.write("\t".join(TASK_KEYS) + "\n" + _tsv(tasks(), TASK_KEYS))
        return 0
    rows = lp.read_manifest(args.manifest)
    enc = encoding.Encoder.fit(encoding.read_covariates(args.covariates), fit_pids(rows))
    tr = ([dict(p) for p in train_pairs(rows)] + xassay_pairs(rows))
    import json
    print(json.dumps(leakage_audit(rows, enc, tr, law_pairs(rows)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
