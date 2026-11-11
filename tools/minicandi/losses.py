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
each kind (absent when a kind has no masked bin).
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


def check_config(config: dict) -> None:
    """Raise ValueError for an unknown `loss` or `loss_weights` key."""
    name = config.get("loss", DEFAULT_LOSS)
    if name not in LOSSES:
        raise ValueError(f"loss {name!r} not in {LOSSES}")
    w = config.get("loss_weights") or {}
    bad = sorted(set(w) - set(WEIGHT_KEYS))
    if bad:
        raise ValueError(f"loss_weights keys {bad} not in {WEIGHT_KEYS}")


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
    if name == DEFAULT_LOSS and unit:
        main = C.nb_nll(out["n"], out["mu"], y, mask)      # the pre-A1 loss, bit for bit
        with torch.no_grad():
            per_m = nb_nll_per(out["n"][mask], out["mu"][mask], y[mask])
    else:
        per_m = PER_BIN[name](out["n"][mask], out["mu"][mask], y[mask])
        k = int(mask.sum())
        if k == 0:
            main = (out["mu"] * 0.0).sum()
        else:
            wk = torch.zeros(4, dtype=per_m.dtype, device=per_m.device)
            for kd, key in KIND_KEYS.items():
                wk[kd] = w[key]
            main = (per_m * wk[kind[mask].long()]).sum() / k
    parts["main"] = float(main.detach())
    with torch.no_grad():
        km = kind[mask]
        for kd, key in KIND_KEYS.items():
            sel = km == kd
            if sel.any():
                parts[key] = float(per_m.detach()[sel].mean())
    loss = main
    if "mu_base" in out:
        if "y_base" not in batch or "c_base" not in batch:
            raise KeyError("a two-step output needs batch['y_base'] and batch['c_base']")
        has_base = batch["c_base"][..., C.LOG2_DEPTH_COL] != 0              # [B, A]
        bmask = mask & has_base[:, None, :]
        base = C.nb_nll(out["n_base"], out["mu_base"], batch["y_base"], bmask)
        parts["base"] = float(base.detach())
        loss = loss + w["base"] * base
    return loss, parts


__all__ = ["CRPS_K_CAP", "DEFAULT_LOSS", "LOSSES", "PER_BIN", "WEIGHT_KEYS", "check_config",
           "nb_crps", "nb_nll_per", "poisson_mean_detached_size_per", "total_loss", "weights"]
