"""miniCANDI training losses (plan pinned interface E, "Losses").

    total_loss(out, batch, config) -> (loss, parts)

`config["loss"]` picks the per-bin loss of the main output `(out["n"], out["mu"])` against
`y_counts` on `contract.loss_mask(batch)`:
  nb_nll                       (default) the NB negative log-likelihood, `contract.nb_nll`
  poisson_mean_detached_size   Poisson NLL of y given mu (the mean part) + NB NLL of y given
                               (mu detached, n) (the size part: no gradient reaches mu from it);
                               the PPML form of the CAC testbed (`ladder.model`, loss "ppml")
  nb_crps                      the CRPS of NB(mean mu, size n) against y, differentiable in both
                               (`nb_crps`, below: an exact sum over a +-10 sd window)
`config["loss_weights"]`: `{"impute", "correct", "same", "base"}`, each default 1. The first three
weight the masked bins of targets of `y_kind` 1, 2 and 3; the main term is the weighted sum over
the masked bins divided by the NUMBER of masked bins (so the default weights give the plain mean).
`base` weights the base leg of two-step correction (decision D15): the NB NLL of `y_base` under
`(out["n_base"], out["mu_base"])` on the bins of `loss_mask` of the tracks that have a base view
(`c_base[..., 0] != 0`). The base leg uses the same bins as the main term, so a same-view target
(kind 3) trains its base leg on hidden bins only, as the copy-leak rule asks. It is the NB NLL
whatever `loss` is (interface E).

A config with neither key (and no two-step output) returns exactly `contract.nb_nll(n, mu, y,
loss_mask)`, the loss of every run before this module: 0 difference.

`parts` (floats, for `log.jsonl`): `main` (the main term), `base` (when the output has a base
leg), and `impute`, `correct`, `same`: the unweighted mean per-bin loss over the masked bins of
each kind (absent when a kind has no masked bin). `main_no_control`: the same mean over the masked
bins of tracks 0 to 34 (the control excluded). It is the loss to compare across control arms:
`control_input = false` turns every control target into kind 1, scored on all bins, so `main`
then averages over a different bin set than the base's (t209 ASF.md). Logging only; no gradient.

Loss arms of stage 3 (plan Amendment 3, section B; pinned interface E, plus the key
`loss_bin_equal` of the foreman amendment). Each key at its default (absent, 0, "none", [], false)
leaves the loss bit-identical to the loss above.
  loss_bin_equal (bool)        replaces the per-bin loss by the NB analogue of Max's equal-weight
                               loss: per masked element
                                 nb_nll(n, mu.detach(), y) + 0.5 * (y - mu)^2 / vbar,
                               vbar = the mean of the predicted NB variance mu + mu^2/n (detached)
                               over the loss-mask bins of one (window, track). The likelihood trains
                               only the dispersion n; the mean gets the same weight 1/vbar on the
                               squared error of every bin of a track. Needs `loss` = nb_nll.
  loss_assay_weight            "none" | "equal" | "learned". The main term becomes a mean over the
                               assays (tracks) with masked bins in the batch of a per-assay term,
                               each built from that assay's mean (kind-weighted) per-bin loss L_a.
                               equal: L_a / L_a.detach(), stateless: its value is 1 and its
                               gradient is that of sum over present assays of log(L_a), divided
                               by their number, so every present assay gets the same gradient
                               scale. No state is kept, so a resumed run equals an uninterrupted
                               one. Because `main` is then always 1, part `assay_mean` logs the
                               unweighted mean of L_a over the present assays (both arms).
                               learned: exp(lw_a) * L_a - lw_a, lw = out["assay_log_w"] [A], i.e.
                               the weighted loss plus -sum(log w) over the present assays, divided
                               by their number.
  loss_shape_weight (float)    adds weight x the multinomial negative log-likelihood of y under
                               p = mu / sum(mu) over the loss-mask bins of each (window, track)
                               (the multinomial coefficient included), summed over (window, track)
                               and divided by the number of masked bins: a per-bin scale, like the
                               main term. part `shape`.
  loss_multiscale ([k, ...])   for each block size k adds the mean NB NLL of the block sums of y
                               under the pooled prediction `pool_nb(mu, n, k)` (moment-matched size
                               under conditional independence, Amendment 3 section C), over the
                               blocks of k consecutive bins that the loss mask covers whole; a
                               trailing L mod k bins are dropped. part `multiscale_<k>`.

Per-window loss weights (pinned interface I, chunk TR). A batch from a `train_regions` sampler
carries `w_window` float32 [B]. Every per-bin loss of window b is then multiplied by w_window[b] in
every term, and every denominator stays as above (the counts of bins, blocks or windows, never the
sum of weights): the main term (each per-bin loss, before the kind weights, the bin-equal form or
the per-assay mean; with `loss_assay_weight` the per-assay sum is weighted and its count is not),
the shape term (each (window, track) multinomial NLL), each multiscale term (each block NLL) and
the base leg (each base bin). With w = genome share / sampling share the expected weighted loss under
the tilted draw equals the loss under the uniform draw. The `parts` stay unweighted per-bin means
(`impute`, `correct`, `same`, `main_no_control`), so they compare across samplers; `main`, `shape`,
`multiscale_<k>` and `base` are the weighted terms. Without `w_window` every term is bit-identical.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder.model import nb_nll as _ladder_nb_nll  # noqa: E402
from ladder.model import poisson_nll as _ladder_poisson_nll  # noqa: E402
from minicandi import contract as C  # noqa: E402

LOSSES = ("nb_nll", "poisson_mean_detached_size", "nb_crps")
DEFAULT_LOSS = "nb_nll"
WEIGHT_KEYS = ("impute", "correct", "same", "base")
KIND_KEYS = {C.KIND_IMPUTE: "impute", C.KIND_CORRECT: "correct", C.KIND_SAME: "same"}
#: CRPS grid: k runs over mean +- CRPS_SD sd, in buckets of power-of-two width (at least 16)
CRPS_SD = 10.0
CRPS_K_MIN = 16
#: an element whose window would pass this many points (sd above ~3000 counts) is scored by the
#: Gaussian CRPS with the NB's mean and sd
CRPS_K_CAP = 65536
ASSAY_WEIGHTS = ("none", "equal", "learned")


def check_config(config: dict) -> None:
    """Raise ValueError for an unknown `loss` or `loss_weights` key."""
    name = config.get("loss", DEFAULT_LOSS)
    if name not in LOSSES:
        raise ValueError(f"loss {name!r} not in {LOSSES}")
    w = config.get("loss_weights") or {}
    bad = sorted(set(w) - set(WEIGHT_KEYS))
    if bad:
        raise ValueError(f"loss_weights keys {bad} not in {WEIGHT_KEYS}")
    aw = config.get("loss_assay_weight", "none")
    if aw not in ASSAY_WEIGHTS:
        raise ValueError(f"loss_assay_weight {aw!r} not in {ASSAY_WEIGHTS}")
    sw = config.get("loss_shape_weight", 0)
    if not isinstance(sw, (int, float)) or isinstance(sw, bool) or sw < 0:
        raise ValueError(f"loss_shape_weight {sw!r} must be a number >= 0")
    ks = config.get("loss_multiscale") or []
    if not all(isinstance(k, int) and not isinstance(k, bool) and k >= 2 for k in ks):
        raise ValueError(f"loss_multiscale {ks!r} must be a list of integers >= 2")
    be = config.get("loss_bin_equal", False)
    if not isinstance(be, bool):
        raise ValueError(f"loss_bin_equal {be!r} must be a bool")
    if be and name != DEFAULT_LOSS:
        raise ValueError(f"loss_bin_equal needs loss {DEFAULT_LOSS!r}, not {name!r}")


def weights(config: dict) -> dict:
    w = {k: 1.0 for k in WEIGHT_KEYS}
    w.update({k: float(v) for k, v in (config.get("loss_weights") or {}).items()})
    return w


# ---------------------------------------------------------------------------------- per bin


def nb_nll_per(n, mu, y):
    """Per-element NB NLL, the same computation and clamps as `contract.nb_nll`."""
    return _ladder_nb_nll(torch.log(mu.clamp_min(C.MU_MIN)), torch.log(n.clamp_min(C.N_MIN)),
                          y.to(mu.dtype))


def poisson_mean_detached_size_per(n, mu, y):
    """Per-element Poisson NLL of y given mu + NB NLL of y given (mu detached, n)."""
    log_mu = torch.log(mu.clamp_min(C.MU_MIN))
    yy = y.to(mu.dtype)
    return (_ladder_poisson_nll(log_mu, yy)
            + _ladder_nb_nll(log_mu.detach(), torch.log(n.clamp_min(C.N_MIN)), yy))


def _gauss_crps(m, sd, y):
    z = (y - m) / sd
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return sd * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def nb_crps(n, mu, y):
    """Per-element CRPS of NB(mean mu, size n) against integer y, float64 inside, mu's dtype out.

    CRPS = sum_{k >= 0} (F(k) - 1[k >= y])^2, F the NB CDF. The sum is evaluated exactly on the
    window k = lo .. lo + W - 1, lo = max(0, floor(mu - 10 sd)), W the smallest power of two (at
    least 16) that reaches mu + 10 sd: there the pmf comes from lgamma in log space and F is its
    cumulative sum from lo. Outside the window F is taken as 0 below and 1 above, so each k between
    y and the window adds exactly 1: + max(0, lo - y) + max(0, y - lo - W). The neglected mass is
    beyond 10 sd (relative error about 1e-6 at n = 0.2, mu = 40). Elements of equal W are
    evaluated together. An element with W > 65536 is scored by the Gaussian CRPS of the same mean
    and sd. mu and n are clamped as in `contract.nb_nll`.
    """
    shape, out_dtype = mu.shape, mu.dtype
    mu_d = mu.reshape(-1).double().clamp(C.MU_MIN, C.MU_MAX)
    n_d = n.reshape(-1).double().clamp(C.N_MIN, C.N_MAX)
    y_d = y.reshape(-1).double().clamp_min(0.0)
    sd = torch.sqrt(mu_d + mu_d * mu_d / n_d)
    with torch.no_grad():
        lo = torch.clamp(torch.floor(mu_d - CRPS_SD * sd), min=0.0)
        span = torch.ceil(mu_d + CRPS_SD * sd) - lo + 2.0
        ee = torch.clamp(torch.ceil(torch.log2(span)), min=math.log2(CRPS_K_MIN))
    out = torch.zeros_like(mu_d)
    big = ee > math.log2(CRPS_K_CAP)
    if big.any():
        out = out.index_put((big.nonzero(as_tuple=True)[0],),
                            _gauss_crps(mu_d[big], sd[big], y_d[big]))
    for e in torch.unique(ee[~big]).tolist():
        idx = ((ee == e) & ~big).nonzero(as_tuple=True)[0]
        W = int(2 ** e)
        m, s, yy, l0 = mu_d[idx, None], n_d[idx, None], y_d[idx, None], lo[idx, None]
        k = l0 + torch.arange(W, dtype=torch.float64, device=mu.device)[None, :]
        log_p = torch.log(s) - torch.log(s + m)                  # log P(success) = n/(n+mu)
        log_q = torch.log(m) - torch.log(s + m)
        logpmf = (torch.lgamma(k + s) - torch.lgamma(s) - torch.lgamma(k + 1.0)
                  + s * log_p + k * log_q)
        F = torch.cumsum(torch.exp(logpmf), dim=1).clamp(max=1.0)
        step = (k >= yy).to(torch.float64)
        outside = (l0 - yy).clamp_min(0.0) + (yy - l0 - W).clamp_min(0.0)
        out = out.index_put((idx,), ((F - step) ** 2).sum(1) + outside[:, 0])
    return out.reshape(shape).to(out_dtype)


PER_BIN = {"nb_nll": nb_nll_per, "poisson_mean_detached_size": poisson_mean_detached_size_per,
           "nb_crps": nb_crps}


# ---------------------------------------------------------------------------------- loss arms


def bin_equal_per(n, mu, y, mask):
    """[B, L, A] element loss of `loss_bin_equal`: nb_nll(n, mu.detach(), y) + 0.5 (y - mu)^2 / vbar,
    vbar [B, 1, A] the detached mean of mu + mu^2/n over the `mask` bins of each (window, track)."""
    yy = y.to(mu.dtype)
    m = mask.to(mu.dtype)
    with torch.no_grad():
        mc = mu.clamp(C.MU_MIN, C.MU_MAX)
        var = mc + mc * mc / n.clamp(C.N_MIN, C.N_MAX)
        vbar = (var * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1.0)
        vbar = vbar.clamp_min(C.MU_MIN)
    nll = _ladder_nb_nll(torch.log(mu.detach().clamp_min(C.MU_MIN)),
                         torch.log(n.clamp_min(C.N_MIN)), yy)
    return nll + 0.5 * (yy - mu) ** 2 / vbar


def pool_nb(mu, n, k: int, dim: int = 1):
    """(mu_pool, n_pool): the NB of the sum of k consecutive bins along `dim` (a trailing
    remainder is dropped). mu_pool = sum mu_i; n_pool = (sum mu_i)^2 / sum(mu_i^2 / n_i), the
    moment-matched size under conditional independence (Amendment 3, section C). mu and n are
    clamped below as in `contract.nb_nll` first."""
    mu = mu.clamp_min(C.MU_MIN).movedim(dim, -1)
    n = n.clamp_min(C.N_MIN).movedim(dim, -1)
    nb = mu.shape[-1] // k
    mu = mu[..., :nb * k].reshape(*mu.shape[:-1], nb, k)
    n = n[..., :nb * k].reshape(*n.shape[:-1], nb, k)
    s = mu.sum(-1)
    v = (mu * mu / n).sum(-1)
    return s.movedim(-1, dim), (s * s / v).movedim(-1, dim)


def _block_sum(x, k: int, dim: int = 1):
    x = x.movedim(dim, -1)
    nb = x.shape[-1] // k
    return x[..., :nb * k].reshape(*x.shape[:-1], nb, k).sum(-1).movedim(-1, dim)


def multiscale_nll(n, mu, y, mask, k: int, w=None):
    """Mean NB NLL of the k-bin sums of y under `pool_nb(mu, n, k)` over the blocks that `mask`
    covers whole; a zero with a graph when there is none. `w` [B]: each block of window b is
    weighted by w[b]; the denominator stays the block count."""
    mu_p, n_p = pool_nb(mu, n, k)
    y_p = _block_sum(y.to(mu.dtype), k)
    full = _block_sum(mask.to(torch.int64), k) == k
    if not full.any():
        return (mu * 0.0).sum()
    if w is None:
        return nb_nll_per(n_p[full], mu_p[full], y_p[full]).mean()
    wf = w.to(mu.dtype)[:, None, None].expand_as(full)[full]
    return (nb_nll_per(n_p[full], mu_p[full], y_p[full]) * wf).sum() / int(full.sum())


def shape_nll(mu, y, mask, w=None):
    """Multinomial NLL of y under mu / sum(mu) over the `mask` bins of each (window, track), summed
    over (window, track) and divided by the number of masked bins. `w` [B]: the NLL of each
    (window b, track) is weighted by w[b]; the denominator stays the masked-bin count."""
    k = int(mask.sum())
    if k == 0:
        return (mu * 0.0).sum()
    m = mask.to(mu.dtype)
    yy = y.to(mu.dtype) * m
    mc = mu.clamp_min(C.MU_MIN)
    log_p = torch.log(mc) - torch.log((mc * m).sum(1, keepdim=True).clamp_min(C.MU_MIN))
    N = yy.sum(1)
    nll = (-(yy * log_p).sum(1) - torch.lgamma(N + 1.0)
           + (torch.lgamma(yy + 1.0) * m).sum(1))
    if w is not None:
        nll = nll * w.to(nll.dtype)[:, None]
    return nll.sum() / k


# ---------------------------------------------------------------------------------- total


def total_loss(out: dict, batch: dict, config: dict | None) -> tuple[torch.Tensor, dict]:
    """(loss, parts) of a model output on a batch (module docstring)."""
    config = config or {}
    check_config(config)
    name = config.get("loss", DEFAULT_LOSS)
    w = weights(config)
    mask = C.loss_mask(batch)
    y = batch["y_counts"]
    unit = all(w[k] == 1.0 for k in ("impute", "correct", "same"))
    parts: dict = {}
    kind = batch["y_kind"][:, None, :].expand_as(mask)
    bin_equal = bool(config.get("loss_bin_equal", False))
    assay_w = config.get("loss_assay_weight", "none")
    track = torch.arange(mask.shape[-1], device=mask.device).expand_as(mask)[mask]
    ww = batch.get("w_window")                               # interface I; None by default
    if name == DEFAULT_LOSS and unit and not bin_equal and assay_w == "none" and ww is None:
        main = C.nb_nll(out["n"], out["mu"], y, mask)      # the pre-A1 loss, bit for bit
        with torch.no_grad():
            per_m = nb_nll_per(out["n"][mask], out["mu"][mask], y[mask])
    else:
        if bin_equal:
            per_m = bin_equal_per(out["n"], out["mu"], y, mask)[mask]
        else:
            per_m = PER_BIN[name](out["n"][mask], out["mu"][mask], y[mask])
        k = int(mask.sum())
        if k == 0:
            main = (out["mu"] * 0.0).sum()
        else:
            wk = torch.zeros(4, dtype=per_m.dtype, device=per_m.device)
            for kd, key in KIND_KEYS.items():
                wk[kd] = w[key]
            per_w = per_m * wk[kind[mask].long()]
            if ww is not None:
                per_w = per_w * ww.to(per_w.dtype)[:, None, None].expand_as(mask)[mask]
            if assay_w == "none":
                main = per_w.sum() / k
            else:
                A = mask.shape[-1]
                s_a = torch.zeros(A, dtype=per_w.dtype, device=per_w.device).index_add(
                    0, track, per_w)
                c_a = torch.zeros(A, dtype=per_w.dtype, device=per_w.device).index_add(
                    0, track, torch.ones_like(per_w))
                present = c_a > 0
                la = s_a / c_a.clamp_min(1.0)
                parts["assay_mean"] = float(la[present].detach().mean())
                if assay_w == "equal":
                    lp = la[present]
                    main = (lp / lp.detach().clamp_min(1e-8)).mean()
                else:
                    if "assay_log_w" not in out:
                        raise KeyError("loss_assay_weight = learned needs out['assay_log_w']")
                    lw = out["assay_log_w"].to(la.dtype)[present]
                    main = (torch.exp(lw) * la[present] - lw).mean()
    parts["main"] = float(main.detach())
    with torch.no_grad():
        km = kind[mask]
        for kd, key in KIND_KEYS.items():
            sel = km == kd
            if sel.any():
                parts[key] = float(per_m.detach()[sel].mean())
        nc = track != C.CONTROL_INDEX
        if nc.any():
            parts["main_no_control"] = float(per_m.detach()[nc].mean())
    loss = main
    sw = float(config.get("loss_shape_weight", 0))
    if sw != 0.0:
        shape = shape_nll(out["mu"], y, mask, ww)
        parts["shape"] = float(shape.detach())
        loss = loss + sw * shape
    for kb in config.get("loss_multiscale") or []:
        ms = multiscale_nll(out["n"], out["mu"], y, mask, int(kb), ww)
        parts[f"multiscale_{int(kb)}"] = float(ms.detach())
        loss = loss + ms
    if "mu_base" in out:
        if "y_base" not in batch or "c_base" not in batch:
            raise KeyError("a two-step output needs batch['y_base'] and batch['c_base']")
        has_base = batch["c_base"][..., C.LOG2_DEPTH_COL] != 0              # [B, A]
        bmask = mask & has_base[:, None, :]
        if ww is None:
            base = C.nb_nll(out["n_base"], out["mu_base"], batch["y_base"], bmask)
        else:
            kb_ = int(bmask.sum())
            per_b = nb_nll_per(out["n_base"], out["mu_base"], batch["y_base"])
            per_b = per_b * ww.to(per_b.dtype)[:, None, None]
            base = per_b[bmask].sum() / kb_ if kb_ else (out["mu_base"] * 0.0).sum()
        parts["base"] = float(base.detach())
        loss = loss + w["base"] * base
    return loss, parts


__all__ = ["ASSAY_WEIGHTS", "CRPS_K_CAP", "DEFAULT_LOSS", "LOSSES", "PER_BIN", "WEIGHT_KEYS",
           "bin_equal_per", "check_config", "multiscale_nll", "nb_crps", "nb_nll_per", "pool_nb",
           "poisson_mean_detached_size_per", "shape_nll", "total_loss", "weights"]
