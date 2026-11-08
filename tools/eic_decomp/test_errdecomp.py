# (Written by Claude for Max 2026-10-31)
"""Synthetic tests of errdecomp.py: every component planted with a known size must be recovered.

Run from the sprint folder: python3 scripts/lib/test_errdecomp.py  (about 30 s). Writes nothing unless given
an output path (then it writes a log of the planted vs recovered numbers there)."""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import errdecomp as E

rng = np.random.default_rng(1)
LOG = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    LOG.append(f"{'PASS' if ok else 'FAIL'}  {name}: got {got:.5f}, planted {want:.5f}, tol {tol}")
    assert ok, LOG[-1]


def smooth(n, width):
    """Gaussian-ish smooth random signal with correlation length about `width` bins (box filter, unit variance)."""
    w = rng.standard_normal(n + width)
    c = np.cumsum(w)
    v = (c[width:] - c[:-width])[:n]
    return (v - v.mean()) / v.std()


# 1. calibration identity and terms
n = 400_000
x = rng.gamma(0.5, 1.0, n)
y = 3 + 2 * x                                   # affine: residual 0, nonlinear 0
t, yc = E.calibrate(x, y)
check("affine: identity error", t["ident_err"], 0, 1e-10)
check("affine: residual", t["residual"], 0, 1e-8)
check("affine: offset + scale = mse", t["offset"] + t["scale"], t["mse"], 1e-8)
y = np.sinh(1.5 * x)                            # monotone, nonlinear: residual ~ 0, nonlinear > 0
t, yc = E.calibrate(x, y)
check("monotone: residual small", t["residual"], 0, 2e-3)
assert t["nonlinear"] > 0.05, t
LOG.append(f"PASS  monotone: nonlinear term {t['nonlinear']:.4f} > 0.05")
# optimal shrinkage: y = E[x | z] has slope ratio 1 and zero offset/scale
z = rng.standard_normal(n); x = z + rng.standard_normal(n); y = 0.5 * x * 0 + z      # E[x|z] = z
t, _ = E.calibrate(x, y)
check("conditional mean: slope ratio", t["slope_ratio"], 1.0, 0.01)
check("conditional mean: scale term", t["scale"], 0, 1e-4)
check("conditional mean: residual = 1 - r^2", t["residual"], 0.5, 0.01)
y = 2 * z                                        # over-dispersed: scale term = (2 sd_z - r sd_x)^2 / var x
t, _ = E.calibrate(x, y)
check("over-dispersed: scale term", t["scale"], (2 - 1.0) ** 2 / 2, 0.01)

# 2. Haar: energy preserved, white noise spreads as 2^-j
L = 1 << 20
v = rng.standard_normal(L)
en = E.energy_by_scale(v)
check("haar: energy preserved", en.sum() / (v @ v), 1.0, 1e-9)
for j in (1, 2, 3):
    check(f"haar: white noise level {j} share", en[j - 1] / en.sum(), 2.0 ** -j, 0.005)
s = np.repeat(rng.standard_normal(L // 256), 256)           # constant on 256-bin blocks
en = E.energy_by_scale(s)
check("haar: 256-bin blocks have no energy below level 9", en[:8].sum() / en.sum(), 0, 1e-12)

v = rng.standard_normal(1 << 16)
comps = [E.band_component(v, idx) for idx in E.BANDS.values()]
check("haar: band components add up to the vector", float(np.abs(sum(comps) - v).max()), 0, 1e-9)
check("haar: band components orthogonal", float(np.dot(comps[0], comps[2])), 0, 1e-8)

# 3. shared / specific with groups, and scale localization
masks = {"c1": np.ones(300_000, bool), "c2": np.r_[np.ones(200_000, bool), np.zeros(5_000, bool), np.ones(100_000, bool)]}
lay = E.Layout(masks)
N = lay.nvalid
B = 24
groups = np.repeat(np.arange(12), 2)                      # 12 terms, 2 tracks each (term-mates share some u)
sh_true = 0.3 * smooth(N, 160)                            # shared error, scale ~ 4 kb
sh_e = (sh_true ** 2).mean()
rows = []
for b in range(B):
    u_term = 0.4 * smooth(N, 40) if b % 2 == 0 else None
    rows.append(None)
term_u = {g: 0.4 * smooth(N, 40) for g in range(12)}      # specific error shared within a term (1 kb scale)
M = np.zeros((B, lay.L), np.float32)
spec_e = []
for b in range(B):
    u = term_u[groups[b]] + 0.5 * rng.standard_normal(N)  # plus white specific error
    spec_e.append((u ** 2).mean())
    M[b] = lay.scatter(sh_true + u)
G = E.gram_by_scale(M) / N
sh, sp = E.shared_specific(G, groups)
check("shared part (sum over scales, mean over tracks)", np.nanmean(sh.sum(0)), sh_e, 0.01)
check("specific part (sum over scales, mean over tracks)", np.nanmean(sp.sum(0)), np.mean(spec_e), 0.01)
# white specific error lives mostly at level 1-3; the smooth shared error at levels >= 6
band_sh = {k: np.nanmean(sh[v].sum(0)) for k, v in E.BANDS.items()}
check("shared error: share in fine band", band_sh["fine (<=200 bp)"] / sh_e, 0, 0.02)
# naive split without excluding term-mates overstates shared
G2 = G.copy()
sh_naive, _ = E.shared_specific(G2, np.arange(B))
assert np.nanmean(sh_naive.sum(0)) > np.nanmean(sh.sum(0)) + 0.005
LOG.append(f"PASS  term-mates excluded: shared {np.nanmean(sh.sum(0)):.4f} vs naive {np.nanmean(sh_naive.sum(0)):.4f}")

# 4. full pipeline: truth, prediction = monotone distortion of (truth - planted error); recover offset, residual
xt = 1 + smooth(N, 30) ** 2
err = 0.4 * smooth(N, 8)
w = xt - err
y = np.log1p(np.exp(w) * 3) + 0.7                 # a strictly increasing distortion of w
t, yc = E.calibrate(xt, y)
check("pipeline: identity", t["ident_err"], 0, 1e-9)
# The residual depends only on the ordering of the prediction: equal to the residual of w itself. It is
# E Var(x | w), below the planted error variance (the best predictor of x from w shrinks towards the mean).
t_w, _ = E.calibrate(xt, w)
check("pipeline: residual invariant to monotone distortion", t["residual"], t_w["residual"], 0.002)
res_planted = np.mean(err ** 2) / xt.var()
assert 0.3 * res_planted < t["residual"] < res_planted, (t["residual"], res_planted)
LOG.append(f"PASS  pipeline: residual {t['residual']:.4f} below planted error variance {res_planted:.4f}")

# 5. classification
xx = rng.gamma(0.3, 1, 200_000)
pos = xx > np.quantile(xx, 0.95)
r = E.threshold_row(pos, E.RankedPrediction(xx))
check("class: perfect auroc", r["auroc"], 1, 1e-12); check("class: perfect auprc", r["auprc"], 1, 1e-12)
r = E.threshold_row(pos, E.RankedPrediction(rng.random(200_000)))
check("class: random auroc", r["auroc"], 0.5, 0.01); check("class: random auprc", r["auprc"], 0.05, 0.005)
# runs: three truth peaks of 10 bins; prediction hits one exactly, one shifted by 2, misses one
masks = {"c": np.ones(2 ** 14, bool)}
lay = E.Layout(masks)
xt = np.zeros(2 ** 14); xt[100:110] = 5; xt[500:510] = 5; xt[900:910] = 5
yp = np.zeros(2 ** 14); yp[100:110] = 4; yp[502:512] = 4; yp[3000:3010] = 4
yp += rng.random(2 ** 14) * 1e-3
r = E.threshold_row(xt > 1, E.RankedPrediction(yp), layout=lay)
assert r["n_truth_runs"] == 3 and r["n_truth_runs_hit"] == 2, r
assert r["FN_in_missed_runs"] == 10 and r["FN_matched"] == 12 and r["FP_in_spurious_runs"] == 10, r
LOG.append("PASS  runs: 3 truth peaks, 2 hit; FN 12 of which 10 in the missed peak; FP 12 of which 10 spurious")

# 6. Spearman-Brown on Poisson counts: two halves -> full-depth reliability
lam = np.exp(1.5 * smooth(300_000, 20))
full1 = rng.poisson(lam); full2 = rng.poisson(lam)
h1 = rng.binomial(full1, 0.5); h2 = full1 - h1
r_half = np.corrcoef(h1, h2)[0, 1]
check("Spearman-Brown: halves -> full reliability", E.split_half_full(r_half), np.corrcoef(full1, full2)[0, 1], 0.01)

print("\n".join(LOG))
print(f"all {len(LOG)} checks passed")
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write("\n".join(LOG) + f"\nall {len(LOG)} checks passed\n")
