"""t141 (q11): per t112 arm product, base and arm on the 1 kb grid, the one-warp, and its residual.

Exploratory (PI, 2026-10-29); readings only, no verdicts.

    python arms.py list <products_dir>                            # the sorted arm pids, one per line
    python arms.py arm  <products_dir> <out_dir> <arm_index>      # -> <out_dir>/arms1k/<pid>.npz

`arm_index` is the arm product's position in the sorted list of NON-base pids of
`<products_dir>/MANIFEST.tsv` (the base products are skipped; `list` prints that list). Its track's
base product is the row with the same `track` and `arm == base`.

The 1 kb grid is the t137/t138 one: every chromosome of the npz except chrM / chrY, 1 kb = 40 x
25 bp, tail dropped; counts are summed over the 40 bins (`floor.sums_1k`), -log10 p averaged (as
`knob_effects.load`). The blacklist is handled as in the t138 map: NOT applied at 1 kb (the map is
genome-wide).

Per space the arm is scored in (space rules below):
  warp  the one-warp: the monotone base -> arm map, QuantileMatching of `baseline_rungs`
        (`qm_fit` on the two 1 kb histograms, `qm_apply`), fitted ONCE genome-wide on every 1 kb bin.
  r     counts: log2((arm + 0.5) / (warp(base) + 0.5)).
        pval:   arm - warp(base), a DIFFERENCE on the -log10 p scale (p is already a log scale).

The depth arm also gets the thinning null, counts only (no thinning is possible in -log10 p without
MACS2 reruns): the base's 25 bp counts thinned binomially to the arm's depth, ratio = arm depth /
base depth (MANIFEST `depth`), rng `np.random.default_rng([seed, pid_index])` for seeds 0 and 1, with
`pid_index` = the ARM product's position in the sorted list of ALL pids of the MANIFEST; chromosomes
thinned in npz order from one rng object per seed (`floor.thin`). The thinned base is then summed to
1 kb, warped from the base and residualised exactly as the arm.

Space rules: ctlid, ctldepth and ratio change only the control, so they are scored in pval only;
every other knob (extsize included) in both; the DNase mapq arms are dropped (a t112 defect: they
are byte-identical to the DNase base) and get a `<pid>.DROPPED` marker instead of an npz.

npz keys, per chromosome c and space s in (counts, pval):
  s_base_c, s_arm_c, s_warp_c (= warp(base)), s_r_c                     float32 (counts base/arm uint32)
  counts_null<k>_c (thinned base, uint32), counts_nullwarp<k>_c, counts_nullr<k>_c    depth arm only
  meta   a JSON string: pid, base pid, track, assay, arm, level, depth, spaces, identical_to_base per
         space, qm_n_knots per fit, thin ratio and seeds, pid_index, formulas.
Written atomically (<pid>.tmp.npz, then renamed).
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
for _d in ("t118", "t137"):
    if str(HERE.parents[1] / _d) not in sys.path:
        sys.path.insert(0, str(HERE.parents[1] / _d))
from knob_effects import COARSE, MARK_CLASS  # noqa: E402
import baseline_rungs as br  # noqa: E402  (imports candi: run under the kit venv)
from floor import sums_1k, thin  # noqa: E402

DROP = ("chrM", "chrY")
CONTROL_P_ONLY = ("ctlid", "ctldepth", "ratio")
SPACE_FILES = {"counts": "counts25.npz", "pval": "pval25.npz"}
NULL_SEEDS = (0, 1)
PSEUDO = 0.5
FORMULAS = {
    "grid": "1 kb = 40 x 25 bp, tail dropped; every chromosome but chrM/chrY; no blacklist",
    "coarse": "counts summed, -log10 p averaged over the 40 bins",
    "warp": "baseline_rungs QuantileMatching (qm_fit / qm_apply), base -> arm, fitted once "
            "genome-wide on every 1 kb bin",
    "r_counts": "log2((arm + 0.5) / (warp(base) + 0.5))",
    "r_pval": "arm - warp(base) (difference on the -log10 p scale)",
    "null": "depth arm, counts only: base 25 bp counts thinned by rng.binomial(counts, depth_arm / "
            "depth_base), rng = np.random.default_rng([seed, pid_index]), seeds 0, 1, chromosomes "
            "in npz order; then summed to 1 kb, warped from the base and residualised as the arm",
}


def read_manifest(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    seen, out = set(), []
    for r in rows:
        if r["pid"] not in seen:
            seen.add(r["pid"])
            out.append(r)
    return sorted(out, key=lambda r: r["pid"])


def arm_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["arm"] != "base"]


def spaces_for(mark_class: str, arm: str) -> tuple:
    """The spaces an arm is scored in: () for DNase mapq, pval only for the control knobs."""
    if mark_class == "DNase" and arm == "mapq":
        return ()
    if arm in CONTROL_P_ONLY:
        return ("pval",)
    return ("counts", "pval")


def coarse(x: np.ndarray, space: str) -> np.ndarray:
    if space == "counts":
        return sums_1k(x)
    m = (x.size // COARSE) * COARSE
    return x[:m].astype(np.float64).reshape(-1, COARSE).mean(1).astype(np.float32)


def load_1k(product_dir: Path, space: str) -> dict:
    with np.load(Path(product_dir) / SPACE_FILES[space], allow_pickle=False) as z:
        return {c: coarse(z[c], space) for c in z.files if c not in DROP}


def fit_warp(src: np.ndarray, tgt: np.ndarray):
    """QuantileMatching knots of the monotone map src -> tgt (baseline_rungs.qm_fit)."""
    sv, sc = np.unique(src, return_counts=True)
    tv, tc = np.unique(tgt, return_counts=True)
    return br.qm_fit(sv, sc, tv, tc)


def residual(target: np.ndarray, warped: np.ndarray, space: str) -> np.ndarray:
    t = target.astype(np.float64)
    if space == "counts":
        return np.log2((t + PSEUDO) / (warped + PSEUDO))
    return t - warped


def warp_and_residual(base: dict, target: dict, chroms: list, space: str):
    """Fit the one-warp base -> target genome-wide; per chromosome warp(base) and r."""
    knots = fit_warp(np.concatenate([base[c] for c in chroms]),
                     np.concatenate([target[c] for c in chroms]))
    warp, r = {}, {}
    for c in chroms:
        w = br.qm_apply(knots, base[c])
        warp[c] = w.astype(np.float32)
        r[c] = residual(target[c], w, space).astype(np.float32)
    return warp, r, int(knots[0].size)


def _git_sha() -> str:
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"]
    try:
        return subprocess.run(["git", "-C", str(HERE.parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def arm(products_dir, out_dir, arm_index: int) -> int:
    products_dir, out_dir = Path(products_dir), Path(out_dir)
    rows = read_manifest(products_dir / "MANIFEST.tsv")
    arms = arm_rows(rows)
    if not 0 <= arm_index < len(arms):
        print(f"arm_index {arm_index} outside 0..{len(arms) - 1}", file=sys.stderr)
        return 2
    a = arms[arm_index]
    pid = a["pid"]
    base = next((r for r in rows if r["track"] == a["track"] and r["arm"] == "base"), None)
    if base is None:
        print(f"{pid}: no base product for track {a['track']}", file=sys.stderr)
        return 2
    mclass = MARK_CLASS[a["assay"]]
    spaces = spaces_for(mclass, a["arm"])
    out = out_dir / "arms1k"
    out.mkdir(parents=True, exist_ok=True)
    if not spaces:
        (out / f"{pid}.DROPPED").write_text("DNase mapq arm: byte-identical to the DNase base "
                                            "(t112 defect); dropped as in knob_effects\n")
        print(f"{pid}: dropped (DNase mapq)", flush=True)
        return 0

    pid_index = [r["pid"] for r in rows].index(pid)
    arrays, meta = {}, {
        "pid": pid, "base_pid": base["pid"], "track": a["track"], "assay": a["assay"],
        "mark_class": mclass, "arm": a["arm"], "level": a["level"], "knob": a.get("knob", ""),
        "depth": a.get("depth", ""), "base_depth": base.get("depth", ""), "spaces": list(spaces),
        "arm_index": arm_index, "pid_index": pid_index, "identical_to_base": {},
        "qm_n_knots": {}, "formulas": FORMULAS, "git_sha": _git_sha(),
        "note": "exploratory readings; no verdicts"}
    chroms = None
    for space in spaces:
        b = load_1k(products_dir / base["pid"], space)
        x = load_1k(products_dir / pid, space)
        if list(b) != list(x) or any(b[c].size != x[c].size for c in b):
            print(f"{pid} {space}: chromosomes or lengths differ from {base['pid']}", file=sys.stderr)
            return 3
        chroms = list(b)
        warp, r, nk = warp_and_residual(b, x, chroms, space)
        meta["identical_to_base"][space] = all(np.array_equal(b[c], x[c]) for c in chroms)
        meta["qm_n_knots"][space] = nk
        for c in chroms:
            arrays[f"{space}_base_{c}"] = b[c]
            arrays[f"{space}_arm_{c}"] = x[c]
            arrays[f"{space}_warp_{c}"] = warp[c]
            arrays[f"{space}_r_{c}"] = r[c]
        print(f"{pid} {space}: knots={nk} median|r|="
              f"{np.median(np.abs(np.concatenate([r[c] for c in chroms]))):.4f}", flush=True)

    if a["arm"] == "depth":
        ratio = float(a["depth"]) / float(base["depth"])
        if not 0 < ratio < 1:
            print(f"{pid}: thin ratio {ratio} not in (0, 1)", file=sys.stderr)
            return 3
        meta["null"] = {"thin_ratio": ratio, "seeds": list(NULL_SEEDS),
                        "rng": "np.random.default_rng([seed, pid_index])", "space": "counts"}
        rngs = {s: np.random.default_rng([s, pid_index]) for s in NULL_SEEDS}
        thinned = {s: {} for s in NULL_SEEDS}
        with np.load(products_dir / base["pid"] / SPACE_FILES["counts"], allow_pickle=False) as z:
            for c in z.files:                       # npz order: the draw is reproducible
                if c in DROP:
                    continue
                x25 = z[c]
                for s in NULL_SEEDS:
                    thinned[s][c] = sums_1k(thin(x25, ratio, rngs[s]))
        b = {c: arrays[f"counts_base_{c}"] for c in chroms}
        for s in NULL_SEEDS:
            warp, r, nk = warp_and_residual(b, thinned[s], chroms, "counts")
            meta["qm_n_knots"][f"null{s}"] = nk
            for c in chroms:
                arrays[f"counts_null{s}_{c}"] = thinned[s][c]
                arrays[f"counts_nullwarp{s}_{c}"] = warp[c]
                arrays[f"counts_nullr{s}_{c}"] = r[c]
            print(f"{pid} null seed {s}: ratio={ratio:.4f} knots={nk}", flush=True)

    meta["chroms"] = chroms
    arrays["meta"] = np.array(json.dumps(meta))
    tmp = out / f"{pid}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out / f"{pid}.npz")
    print(f"done {pid}: spaces {spaces}, {len(arrays) - 1} arrays", flush=True)
    return 0


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) == 2 and a[0] == "list":
        for r in arm_rows(read_manifest(Path(a[1]) / "MANIFEST.tsv")):
            print(r["pid"])
        return 0
    if len(a) == 4 and a[0] == "arm":
        return arm(a[1], a[2], int(a[3]))
    print("usage: arms.py list <products_dir> | arms.py arm <products_dir> <out_dir> <arm_index>",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
