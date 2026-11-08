# (Written by Claude for Max 2026-10-31)
"""Exact decomposition of imputation error (sprint 06).

Numpy only. All functions work on one target track at a time, except the shared/specific split, which
works on the Gram matrices of several tracks of one assay. Notation: x truth, y prediction, a training
average, all 1-D float arrays over the valid 25 bp bins of the evaluation chromosomes (concatenated),
in one unit (asinh-p in this project).

The decomposition of one track's mean squared error, in units of the truth's variance Var(x):

  MSE(y) / Var(x) = offset + scale + nonlinear + residual
    offset    = (mean x - mean y)^2 / Var(x)                 fixed by adding a constant
    scale     = (SD y - r SD x)^2 / Var(x)                    fixed by the best affine map (with offset)
    nonlinear = (affine residual - monotone residual)/Var(x) fixed by the best monotone map beyond affine
    residual  = MSE(f(y)) / Var(x), f the best monotone map  ("ranking error": needs a reordering of bins)
  (offset + scale + Var(x)(1 - r^2) = MSE(y) exactly; r = Pearson correlation of x and y.)

The residual e = x - f(y), standardized by SD(x), is then split exactly over spatial scales by an
orthonormal Haar wavelet transform (energy at block sizes 50 bp .. 409.6 kb plus the coarse remainder),
and, across the held-out tracks of one assay, into a part shared by all tracks at a position and a
track-specific part (`shared_specific`), using the Gram matrices of the standardized residuals at each
scale.
"""
import numpy as np

J = 14              # Haar levels: detail j compares the two halves of blocks of 2^j bins (25 * 2^j bp)
BIN_BP = 25


def scale_labels(J=J, bin_bp=BIN_BP):
    """Block size in bp of each Haar level 1..J, then 'coarse' (block means at 2^J bins)."""
    out = []
    for j in range(1, J + 1):
        bp = bin_bp * 2 ** j
        out.append(f"{bp} bp" if bp < 1000 else f"{bp / 1000:g} kb")
    return out + [f">{bin_bp * 2 ** J / 1000:g} kb"]


# Bands used in the report (indices into the J + 1 scales: level j is index j - 1, coarse is index J).
BANDS = {"fine (<=200 bp)": [0, 1, 2],             # 50, 100, 200 bp
         "peak (0.4-3.2 kb)": [3, 4, 5, 6],         # 400 bp .. 3.2 kb
         "domain (6.4-51 kb)": [7, 8, 9, 10],       # 6.4 .. 51.2 kb
         "broad (>=102 kb)": [11, 12, 13, 14]}      # 102.4, 204.8, 409.6 kb and coarse


class Layout:
    """Genome layout of an evaluation set. masks: dict chrom -> boolean array of valid 25 bp bins (full chromosome).

    Valid-bin vectors are the concatenation over chromosomes (in dict order) of the valid bins. A padded
    vector holds every bin of every chromosome, each chromosome padded with zeros to a multiple of 2^J bins,
    so a Haar transform of the padded concatenation is exactly the per-chromosome transform. Invalid bins and
    padding hold 0, so they add nothing to any energy or inner product."""

    def __init__(self, masks, J=J):
        self.J, self.chroms = J, list(masks)
        blk = 2 ** J
        idx, off, self.seg = [], 0, {}
        for c in self.chroms:
            m = np.asarray(masks[c], bool)
            n = len(m); npad = -(-n // blk) * blk
            idx.append(off + np.flatnonzero(m))
            self.seg[c] = (off, n, int(m.sum()))
            off += npad
        self.L = off
        self.idx = np.concatenate(idx).astype(np.int64)
        self.nvalid = len(self.idx)

    def scatter(self, v, dtype=np.float32):
        full = np.zeros(self.L, dtype)
        full[self.idx] = v
        return full


def pav(v, w):
    """Weighted pool-adjacent-violators: the non-decreasing sequence closest to v in weighted least squares."""
    vals, wts, cnt = [], [], []
    for vi, wi in zip(v.tolist(), w.tolist()):
        vals.append(vi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w2 = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w2
            c2 = cnt[-2] + cnt[-1]
            vals[-2:] = [v2]; wts[-2:] = [w2]; cnt[-2:] = [c2]
    return np.repeat(np.array(vals), cnt)


def monotone_fit(y, x, n_body=1600, n_tail=400):
    """Best non-decreasing function of y for predicting x: isotonic regression of the bin means of x on quantile
    bins of y (n_body bins over quantiles 0-0.99, n_tail over 0.99-1, so the sparse top is resolved), linearly
    interpolated between bin means of y. Returns (knots, values) for np.interp."""
    qs = np.concatenate([np.linspace(0, .99, n_body, endpoint=False), np.linspace(.99, 1, n_tail + 1)])
    edges = np.unique(np.quantile(y, qs))
    if len(edges) < 2:
        return np.array([y.min()]), np.array([x.mean()])
    k = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, len(edges) - 2)
    cnt = np.bincount(k, minlength=len(edges) - 1).astype(np.float64)
    sx = np.bincount(k, weights=x, minlength=len(edges) - 1)
    sy = np.bincount(k, weights=y, minlength=len(edges) - 1)
    keep = cnt > 0
    my, mx, w = sy[keep] / cnt[keep], sx[keep] / cnt[keep], cnt[keep]
    return my, pav(mx, w)


def calibrate(x, y):
    """Calibration terms of y as a prediction of x (see module docstring), and the calibrated prediction f(y).

    Returns (terms, yc). terms holds, in units of Var(x): mse, offset, scale, nonlinear, residual (they add up
    exactly: mse = offset + scale + nonlinear + residual); plus r (Pearson), slope_ratio = SD(y) / (r SD(x))
    (1 = the optimal amount of regression to the mean; < 1 over-shrunk; > 1 over-dispersed), mono_used
    (False if the monotone fit was no better than the affine one, so the affine map is used), var_x, mean_x."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    sx, sy = np.sqrt(vx), np.sqrt(vy)
    r = cov / (sx * sy) if sy > 0 else 0.0
    mse = np.mean((x - y) ** 2)
    offset = (mx - my) ** 2
    scale = (sy - r * sx) ** 2
    aff = vx * (1 - r ** 2)
    sign = 1.0 if r >= 0 else -1.0
    kn, val = monotone_fit(sign * y, x)
    yc = np.interp(sign * y, kn, val)
    mono = np.mean((x - yc) ** 2)
    used = True
    if not mono < aff:              # the affine map is itself monotone, so the best monotone fit is never worse
        yc = mx + (cov / vy) * (y - my) if vy > 0 else np.full_like(x, mx)
        mono, used = aff, False
    t = dict(mse=mse / vx, offset=offset / vx, scale=scale / vx, nonlinear=(aff - mono) / vx, residual=mono / vx,
             r=r, slope_ratio=(sy / (r * sx)) if r > 0 else np.nan, mono_used=used, var_x=vx, mean_x=mx,
             ident_err=(mse - offset - scale - aff) / vx)
    return t, yc


def haar(M, J=J):
    """Orthonormal Haar transform along the last axis (length a multiple of 2^J). Returns [D_1, ..., D_J, A_J]:
    detail coefficients of each level and the coarse approximation. Sum of squares is preserved exactly."""
    s = np.float32(np.sqrt(0.5)) if M.dtype == np.float32 else np.sqrt(0.5)
    out, A = [], M
    for _ in range(J):
        a, b = A[..., 0::2], A[..., 1::2]
        out.append((a - b) * s)
        A = (a + b) * s
    out.append(A)
    return out


def haar_inverse(coefs):
    """Inverse of haar(): coefficients [D_1, ..., D_J, A_J] back to the vector."""
    s = np.sqrt(0.5)
    A = coefs[-1]
    for D in coefs[-2::-1]:
        out = np.empty(A.shape[:-1] + (2 * A.shape[-1],), dtype=np.result_type(A, D))
        out[..., 0::2] = (A + D) * s
        out[..., 1::2] = (A - D) * s
        A = out
    return A


def band_component(v_full, idx, J=J):
    """The part of a padded vector carried by the Haar scales in idx (indices 0..J; J = coarse), as a vector.
    The components of disjoint index sets are orthogonal and add up to v_full."""
    c = haar(np.asarray(v_full, np.float64), J)
    return haar_inverse([ck if k in idx else np.zeros_like(ck) for k, ck in enumerate(c)])


def energy_by_scale(v_full, J=J):
    """Energy (sum of squares) of a padded full vector at each Haar level 1..J and the coarse remainder."""
    return np.array([float(np.dot(c.astype(np.float64), c.astype(np.float64))) for c in haar(v_full, J)])


def gram_by_scale(M, J=J, chunk=1 << 22):
    """M: (B, L) padded full vectors of B tracks. Returns G (J + 1, B, B): inner products of the tracks' Haar
    coefficients at each scale (float64 accumulation). G[:, b, b].sum() == ||M[b]||^2."""
    coefs = haar(M, J)
    B = M.shape[0]
    G = np.zeros((len(coefs), B, B))
    for k, C in enumerate(coefs):
        for s0 in range(0, C.shape[1], chunk):
            c = C[:, s0:s0 + chunk].astype(np.float64)
            G[k] += c @ c.T
    return G


def shared_specific(G, groups):
    """Split each track's energy at each scale into a shared and a specific part.

    G: (K, B, B) Gram matrices by scale of B tracks' standardized residuals (one assay). groups: length-B labels
    (ENCODE term): tracks of the same group are never used to estimate each other's shared part.
    Model: e_b = s + u_b, s common to all tracks, u_b independent across groups with mean 0. Then
    E<e_b, e_c> = ||s||^2 for b, c in different groups, so shared_b = mean over such c of G[k, b, c] is unbiased
    for ||s||^2 at scale k, and specific_b = G[k, b, b] - shared_b. Returns (shared, specific), each (K, B); NaN
    for a track with no track of another group."""
    groups = np.asarray(groups)
    K, B, _ = G.shape
    sh = np.full((K, B), np.nan)
    for b in range(B):
        o = groups != groups[b]
        if o.any():
            sh[:, b] = G[:, b, o].mean(-1)
    tot = np.einsum("kbb->kb", G)
    return sh, tot - sh


def split_half_full(r_half):
    """Spearman-Brown: reliability of a track made of both halves, from the correlation of the two halves."""
    r = np.asarray(r_half, float)
    return 2 * r / (1 + r)


# ---------------------------------------------------------------------------------------------------------
# Error as a function of value

QEDGES = np.array([0, .5, .8, .9, .95, .99, .999, 1.0])


def squant(sorted_v, q):
    """Quantiles of a sorted array (lower interpolation-free: the value at index floor(q (n - 1)))."""
    q = np.asarray(q, float)
    return sorted_v[np.floor(q * (len(sorted_v) - 1)).astype(np.int64)]


def strata_sums(v, key, qedges=QEDGES, sorted_key=None):
    """Sum of v and counts within quantile strata of key (strata by within-track quantile: ties go to the
    lower stratum). Returns (sums, counts), each len(qedges) - 1."""
    cut = np.quantile(key, qedges[1:-1]) if sorted_key is None else squant(sorted_key, qedges[1:-1])
    k = np.searchsorted(cut, key, side="left")
    n = len(qedges) - 1
    return np.bincount(k, weights=v, minlength=n), np.bincount(k, minlength=n).astype(float)


def cond_curve(key, val, nbin=40, sorted_key=None):
    """Mean of key and of val in nbin quantile bins of key (plus the top 1% split in 4): a conditional-mean curve."""
    qs = np.concatenate([np.linspace(0, .99, nbin, endpoint=False), [.9925, .995, .9975, 1.0]])
    edges = np.quantile(key, qs) if sorted_key is None else squant(sorted_key, qs)
    k = np.clip(np.searchsorted(edges, key, side="right") - 1, 0, len(edges) - 2)
    c = np.bincount(k, minlength=len(edges) - 1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.bincount(k, weights=key, minlength=len(edges) - 1) / c,
                np.bincount(k, weights=val, minlength=len(edges) - 1) / c, c)


# ---------------------------------------------------------------------------------------------------------
# Classification framing: is x > v?

def runs(full_bool):
    """Start and end (exclusive) indices of runs of True in a boolean vector."""
    d = np.diff(np.concatenate([[0], full_bool.view(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def dilate(full_bool, k):
    """True within k bins of any True."""
    if k == 0:
        return full_bool
    c = np.concatenate([[0], np.cumsum(full_bool, dtype=np.int64)])
    n = len(full_bool)
    i = np.arange(n)
    return (c[np.minimum(i + k + 1, n)] - c[np.maximum(i - k, 0)]) > 0


def run_hits(runs_se, other_dil):
    """For each run [s, e), does other_dil have any True inside it?"""
    s, e = runs_se
    c = np.concatenate([[0], np.cumsum(other_dil, dtype=np.int64)])
    return (c[e] - c[s]) > 0


class RankedPrediction:
    """A prediction sorted once (descending, random tie-break), reused for every threshold of a sweep."""

    def __init__(self, y, seed=0):
        rng = np.random.default_rng(seed)
        self.y = np.asarray(y, np.float64)
        # random tie-break: a jitter far below the spacing of distinct values (predictions are float16-derived)
        jit = rng.random(len(y)) * 1e-9 * max(1.0, float(np.abs(self.y).max()))
        self.order = np.argsort(-(self.y + jit), kind="stable")
        self.n = len(y)


def threshold_row(xpos, rp, layout=None, common=None, common_cut=(0.1, 0.5), naive_pred=None, k_peak=4):
    """Classification of the bins with x above a threshold (xpos: boolean, the positives) by prediction rp.

    Counts: P (positives), naive errors (FN, FP of naive_pred, the prediction above the same cutoff as the truth,
    if given), matched-count errors (predict the top P bins of y: then FN = FP), best-threshold errors (the
    cutoff on y minimizing FN + FP), AUPRC (average precision) and AUROC. With common (fraction of training
    tracks positive at each bin): recall of positives and false-positive rate of negatives in three groups
    (rare < 0.1 <= intermediate < 0.5 <= constitutive). With layout: truth runs (peaks) and predicted runs, the
    share of FN bins inside wholly missed truth runs (no predicted positive within k_peak bins) and of FP bins in
    wholly spurious predicted runs."""
    n = rp.n
    P = int(xpos.sum())
    r = {"P": P, "frac_pos": P / n}
    if P == 0 or P == n:
        return r
    lab = xpos[rp.order]
    tp = np.cumsum(lab, dtype=np.int64)
    k = np.arange(1, n + 1)
    # matched count
    TPm = int(tp[P - 1])
    r.update(FN_matched=P - TPm, FP_matched=P - TPm, recall_matched=TPm / P)
    # best threshold (cut after any position; ignoring ties in y)
    err = P + k - 2 * tp
    kb = int(np.argmin(err)); r.update(err_best=int(min(err.min(), P)), k_best=kb + 1)
    if naive_pred is not None:
        FNn = int((xpos & ~naive_pred).sum()); FPn = int((~xpos & naive_pred).sum())
        r.update(FN_naive=FNn, FP_naive=FPn, pred_pos_naive=int(naive_pred.sum()))
    # AUPRC (average precision) and AUROC (ties in y broken at random)
    prec = tp / k
    r["auprc"] = float(prec[lab].sum() / P)
    ranks_pos = np.flatnonzero(lab) + 1                   # 1 = highest prediction
    N = n - P
    r["auroc"] = float(1 - (ranks_pos.sum() - P * (P + 1) / 2) / (P * N))
    predm = np.zeros(n, bool); predm[rp.order[:P]] = True
    if common is not None:
        g = np.digitize(common, common_cut)               # 0 rare, 1 intermediate, 2 constitutive
        for gi, gname in enumerate(["rare", "inter", "const"]):
            m = g == gi
            pg = xpos & m; ng = ~xpos & m
            r[f"P_{gname}"] = int(pg.sum()); r[f"N_{gname}"] = int(ng.sum())
            r[f"TP_{gname}"] = int((pg & predm).sum()); r[f"FP_{gname}"] = int((ng & predm).sum())
    if layout is not None:
        tf = np.zeros(layout.L, bool); tf[layout.idx] = xpos
        pf = np.zeros(layout.L, bool); pf[layout.idx] = predm
        tr, pr = runs(tf), runs(pf)
        thit = run_hits(tr, dilate(pf, k_peak))
        phit = run_hits(pr, dilate(tf, k_peak))
        tlen, plen = tr[1] - tr[0], pr[1] - pr[0]
        fn_full = (tf & ~pf)
        fp_full = (pf & ~tf)
        cfn = np.concatenate([[0], np.cumsum(fn_full, dtype=np.int64)])
        cfp = np.concatenate([[0], np.cumsum(fp_full, dtype=np.int64)])
        fn_in_missed = int((cfn[tr[1]] - cfn[tr[0]])[~thit].sum())
        fp_in_spur = int((cfp[pr[1]] - cfp[pr[0]])[~phit].sum())
        r.update(n_truth_runs=len(tlen), n_truth_runs_hit=int(thit.sum()), truth_bins_in_hit_runs=int(tlen[thit].sum()),
                 n_pred_runs=len(plen), n_pred_runs_hit=int(phit.sum()),
                 FN_in_missed_runs=fn_in_missed, FP_in_spurious_runs=fp_in_spur)
    return r
