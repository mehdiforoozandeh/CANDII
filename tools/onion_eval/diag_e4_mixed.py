"""Close-out E4 review item 2: why the mixed model and the ladder disagree at the fine band (diagnostic).

Reads the counts of one assay exactly as `mixed_model.run` does (same observations, same common depth,
same thinning seeds, same Haar band) and reports, from the same sufficient statistics S = sum_k y_k y_k':
  1. moment pair values 0.5 (S_ii + S_jj - 2 S_ij) / K by pair type (halves of one library; two libraries of
     one experiment; two experiments of one cell; two cells), each the exact per-coefficient analogue of a
     ladder pair value (the per-coefficient mean cancels in a difference);
  2. the ladder's own pair values of the same assay and band from a `layers.py --pairs-out` file;
  3. REML fits: the stored fit (mixed_model.decompose, no bootstrap) and fits from other starting points,
     with the scaled objective and gradient, and whether a component sits at the lower bound;
  4. a moment (ANOVA-type) decomposition from the pair-type means.

CLI: python tools/onion_eval/diag_e4_mixed.py --tables <dir> --evalset <json> --assay <A> --band fine
     --pairs <pair_values.tsv.gz> --out <diag.json> [--second-experiments]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

for _TOOLS in (Path(__file__).resolve().parents[1], Path.cwd() / "tools"):   # in a repo, or run from a kit
    if (_TOOLS / "onion_eval").is_dir() and str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))

from onion_eval import ladder as LD  # noqa: E402
from onion_eval import mixed_model as MM  # noqa: E402

BANDS = MM.BANDS


def pair_type(a, b):
    if a["measurement_id"] == b["measurement_id"]:
        return "halves"
    if a["experiment"] == b["experiment"]:
        return "library"
    if a["cell"] == b["cell"]:
        return "experiment"
    return "cell"


def _unit(o):
    return "pooled" if o["library"].endswith("|pooled") else "br"


def moment_pairs(obs, S, K):
    """Pair values by pair type, by unit types of the two sides (between cells), and per cell (mean
    over the cell's between-cell pairs)."""
    out, units, per_cell = {}, {}, {}
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            t = pair_type(obs[i], obs[j])
            v = 0.5 * (S[i, i] + S[j, j] - 2 * S[i, j]) / K
            out.setdefault(t, []).append(v)
            if t == "cell":
                units.setdefault("-".join(sorted((_unit(obs[i]), _unit(obs[j])))), []).append(v)
                for o in (obs[i], obs[j]):
                    per_cell.setdefault(o["cell"] + ":" + _unit(o), []).append(v)
    res = {t: dict(n=len(v), mean=float(np.mean(v))) for t, v in out.items()}
    res["cell_by_units"] = {t: dict(n=len(v), mean=float(np.mean(v))) for t, v in units.items()}
    res["cell_by_cell"] = {c: round(float(np.mean(v)), 6) for c, v in sorted(per_cell.items())}
    return res


def fit_report(S, ysum, K, labels, factors, theta0=None):
    est = MM.estimability(labels, order=factors)
    fit_f = [f for f in factors if est[f][0]]
    Zs = [MM._incidence(labels[f]) for f in fit_f]
    sig, _, theta = MM.reml_fit(S, ysum, K, Zs, None, theta0=theta0)
    ZZ = [Z @ Z.T for Z in Zs]
    n = S.shape[0]
    v, g = MM._neg2_reml(theta, ZZ, S, ysum, None, K)
    return dict(factors=fit_f + ["residual"], sigma=[float(x) for x in sig],
                fraction=[float(x) for x in sig / sig.sum()], neg2reml=float(v),
                scaled_obj=float(v / (n * K)), scaled_grad=[float(x) for x in g / (n * K)],
                at_lower_bound=[f for f, t in zip(fit_f + ["residual"], theta) if t <= -29.9])


def same_pair_two_ways(tables, regions, assay, band, d_ladder, d_mixed, n=12):
    """Mode --same-pairs: on n uniform/uniform pooled other_cell_type pairs of the assay, the pair value per
    coefficient (a) by layers.pair_variance at the ladder's depth and (b) by the mixed model's response
    (thin each track with the mixed-model seed to its depth, log1p, Haar band, 0.5 mean (ya - yb)^2);
    and the same two ways for the pr1/pr2 halves of each track of those pairs (sampling)."""
    import zlib
    from eic_decomp import errdecomp
    from onion_eval import layers as LY
    meas = {m["measurement_id"]: m for m in MM._read_tsv(Path(tables) / "measurements.tsv")}
    pairs = [p for p in MM._read_tsv(Path(tables) / "pairs.tsv") if p["assay"] == assay and
             p["rung"] == "other_cell_type" and meas[p["a"]]["layer"] == meas[p["b"]]["layer"] == "uniform"
             and meas[p["a"]]["unit"] == meas[p["b"]]["unit"] == "pooled"][:n]
    rd = LY.CountsReader()
    L = 2 ** MM.J
    idx = BANDS[band]
    def load(path):
        return np.stack([rd.read(path, c, s0, s0 + L) for c, s0 in regions])
    def mixed_y(x, depth, path):
        g = np.random.default_rng(zlib.crc32(f"mm:{path}".encode()))
        xx = LY.thin(x, d_mixed / depth, g) if d_mixed < depth else x
        c = errdecomp.haar(np.log1p(np.asarray(xx, np.float64)), MM.J)
        return np.concatenate([c[i].reshape(-1) for i in idx])
    out = []
    for p in pairs:
        ma, mb = meas[p["a"]], meas[p["b"]]
        xa, xb = load(ma["counts_path"]), load(mb["counts_path"])
        da, db = float(ma["depth_reads"]), float(mb["depth_reads"])
        lad = LY.pair_variance(xa, xb, da, db, depth=d_ladder, seed=zlib.crc32(p["pair_id"].encode()))
        ya, yb = mixed_y(xa, da, ma["counts_path"]), mixed_y(xb, db, mb["counts_path"])
        K = ya.size
        nb = xa.size
        row = dict(pair=p["pair_id"], cells=[ma["cell"], mb["cell"]], depths=[da, db],
                   ladder=0.5 * lad["variance"]["all"][band] * nb / K, mixed=0.5 * float(np.mean((ya - yb) ** 2)))
        hal = LY.half_depths(ma)
        if hal:
            (p1, d1), (p2, d2) = hal
            h1, h2 = load(p1), load(p2)
            lh = LY.pair_variance(h1, h2, d1, d2, depth=d_ladder, seed=1)
            y1, y2 = mixed_y(h1, da / 2.0, p1), mixed_y(h2, da / 2.0, p2)
            row.update(halves_attr_depths=[d1, d2], ladder_halves=0.5 * lh["variance"]["all"][band] * nb / K,
                       mixed_halves=0.5 * float(np.mean((y1 - y2) ** 2)))
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--assay", required=True)
    ap.add_argument("--band", default="fine")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--second-experiments", action="store_true")
    ap.add_argument("--same-pairs", action="store_true")
    ap.add_argument("--moments-only", action="store_true")
    ap.add_argument("--d-ladder", type=float, default=0.0)
    ap.add_argument("--d-mixed", type=float, default=0.0)
    a = ap.parse_args(argv)
    ev = json.loads(Path(a.evalset).read_text())
    regions = [(c, int(s)) for c, s in ev["regions"]]
    if a.same_pairs:
        res = same_pair_two_ways(a.tables, regions, a.assay, a.band, a.d_ladder, a.d_mixed)
        Path(a.out).write_text(json.dumps(res, indent=1))
        return 0
    meas = MM._read_tsv(Path(a.tables) / "measurements.tsv")
    obs, _ = MM.select_observations(meas, a.assay, None, ("lab",), a.second_experiments)
    D = MM.common_depth(obs)
    S, ysum, K = MM.sufficient_stats(obs, regions, a.band, depth=D)
    res = dict(assay=a.assay, band=a.band, n_obs=len(obs), K=int(K), common_depth=D,
               moment_pairs=moment_pairs(obs, S, K))
    # the ladder's pair values of the same assay and band (stratum all), per-bin -> per-coefficient x n/K
    lad = {}
    for r in LD._read_tsv(a.pairs):
        if r["assay"] == a.assay and r["stratum"] == "all" and r["band"] == a.band and r["var_d"]:
            lad.setdefault(r["rung"], []).append(0.5 * float(r["var_d"]))
    nb = len(regions) * 2 ** MM.J
    res["ladder_pairs_per_coefficient"] = {k: dict(n=len(v), mean=float(np.mean(v)) * nb / K)
                                           for k, v in lad.items()}
    if a.moments_only:
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return 0
    labels = {f: [o[f] for o in obs] for f in MM.FACTORS}
    labels["lab"] = [o["lab"] for o in obs]
    full = MM.FACTORS + ["lab"]
    res["reml_stored_start"] = fit_report(S, ysum, K, labels, full)
    res["reml_cell_only"] = fit_report(S, ysum, K, labels, ["cell"])
    res["reml_cell_library"] = fit_report(S, ysum, K, labels, ["cell", "library"])
    # start the full fit at moment values: everything between cells in cell, the rest split
    mp = res["moment_pairs"]
    h = mp["halves"]["mean"]
    lib = max(mp.get("library", {"mean": h})["mean"] - h, 1e-8)
    cel = max(mp["cell"]["mean"] - mp.get("library", {"mean": h})["mean"], 1e-8)
    est = MM.estimability(labels, order=full)
    fit_f = [f for f in full if est[f][0]]
    th0 = [math.log(cel if f == "cell" else lib if f == "library" else 1e-4 * h) for f in fit_f] + [math.log(h)]
    res["reml_moment_start"] = fit_report(S, ysum, K, labels, full, theta0=np.array(th0))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
