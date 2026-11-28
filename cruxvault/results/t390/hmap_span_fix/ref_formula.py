"""Deterministic outputs of the t125 hmap depth-offset code on fixed synthetic tracks.

    python ref_formula.py <out.npz> [--span N] [--repo DIR]

`--span N` monkeypatches `hmap.formula.SPAN_EXTRA` and `hmap.caller.SPAN_EXTRA` to N after import
(the bit-exact check sets N = 24 on the new code). Every array the constant reaches is saved:
lambda_track, pileup, P("bin"), P("frag"), P_enumerated and data.formula_log1p_t (CPU) on a k grid.
`--repo DIR` imports hmap from DIR/tools (e.g. an extracted copy of the pre-fix tree); default: this repo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
if "--repo" in sys.argv:
    REPO = Path(sys.argv[sys.argv.index("--repo") + 1]).resolve()
for p in (REPO / "tools" / "t118", REPO / "tools" / "t125"):
    sys.path.insert(0, str(p))

from hmap import caller, data, formula  # noqa: E402

L = 20_000
K = 64
INFOS = {
    "chip_ctl_rl101": dict(assay="H3K27ac", depth_reads=30e6, read_length=101, fraglen=180,
                           has_control=1, control_reads=40e6, ratio_V=None, control_read_length=101.0),
    "chip_ctl_q025": dict(assay="H3K27ac", depth_reads=30e6, read_length=101, fraglen=180,
                          has_control=1, control_reads=10e6, ratio_V=None, control_read_length=101.0),
    "chip_ratio_V056": dict(assay="H3K4me3", depth_reads=30e6, read_length=101, fraglen=200,
                            has_control=1, control_reads=107e6, ratio_V=0.56, control_read_length=76.0),
    "chip_ctl_rl36": dict(assay="H3K27me3", depth_reads=20e6, read_length=36, fraglen=220,
                          has_control=1, control_reads=31e6, ratio_V=None, control_read_length=50.3),
    "dnase_noctl_rl76": dict(assay="DNase-seq", depth_reads=50e6, read_length=76, fraglen=150,
                             has_control=0, control_reads=0, ratio_V=None, control_read_length=0.0),
}


def tracks(seed: int):
    rng = np.random.default_rng(seed)
    rate = np.full(L, 1.2)
    for c in rng.integers(0, L, 60):                       # 60 peaks, 8-40 bins wide, 5-60x
        w = int(rng.integers(8, 40))
        rate[c:c + w] *= float(rng.uniform(5, 60))
    t = rng.poisson(rng.gamma(2.0, rate / 2.0)).astype(np.float64)
    c = rng.poisson(rng.gamma(4.0, np.full(L, 1.6) / 4.0) * (1 + 3 * (rng.random(L) < 0.01))).astype(np.float64)
    return t, c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--span", type=int, default=None)
    ap.add_argument("--repo", default=None)
    a = ap.parse_args()
    if a.span is not None:
        formula.SPAN_EXTRA = a.span
        caller.SPAN_EXTRA = a.span
    out = {"span_extra_formula": np.array(formula.SPAN_EXTRA), "span_extra_caller": np.array(caller.SPAN_EXTRA)}
    for i, (name, kw) in enumerate(INFOS.items()):
        inf = SimpleNamespace(pid=name, **kw)
        t, c = tracks(i)
        cc = c if inf.has_control else None
        lam = formula.lambda_track(inf, cc, t)
        out[f"{name}/lam"] = lam
        out[f"{name}/pileup"] = formula.pileup(t, inf)
        out[f"{name}/P_bin"] = formula.P("bin", inf, t, cc)
        out[f"{name}/P_frag"] = formula.P("frag", inf, t, cc)
        idx = np.arange(0, L, 97)                           # 207 bins x k = 0..K-1
        k = np.broadcast_to(np.arange(K, dtype=np.float64), (idx.size, K))
        rest = formula.frag_rest(t, inf)[idx]
        out[f"{name}/Penum_bin"] = formula.P_enumerated("bin", inf, k, lam[idx])
        out[f"{name}/Penum_frag"] = formula.P_enumerated("frag", inf, k, lam[idx], rest)
        kt = torch.from_numpy(np.ascontiguousarray(k))
        out[f"{name}/log1p_t_bin"] = data.formula_log1p_t("bin", inf, kt, lam[idx]).numpy()
        out[f"{name}/log1p_t_frag"] = data.formula_log1p_t("frag", inf, kt, lam[idx],
                                                           torch.from_numpy(rest)).numpy()
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {len(out)} arrays, SPAN_EXTRA formula={formula.SPAN_EXTRA} "
          f"caller={caller.SPAN_EXTRA}, hmap from {Path(formula.__file__).parent}")


if __name__ == "__main__":
    main()
