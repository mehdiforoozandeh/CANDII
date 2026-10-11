"""t125 part 2: the learned h — models, the feature standardiser, the losses, the checkpoint.

h maps the features of one 25 bp bin (P2a's `features`: rung 1 = 9 columns, rung 5 = 19, the
formula variants one more, log1p of the formula's P, last) to mu = predicted log(1 + P).

Variants (design note Part 2; PI 2026-10-11: MLP 2 x 64, as the t118 g):
- `affine`:          mu = w . feat + b                         (w = 0, b = 0 at init)
- `mlp`:             Linear(F, 64)-ReLU-Linear(64, 64)-ReLU-Linear(64, 1|2), last layer zero
- `formula_affine`:  mu = f + alpha_assay + beta * f            (f = feat[..., -1]; alpha, beta 0)
- `formula_mlp`:     mu = f + MLP(feat)[0]                     (last layer zero)
The formula variants are the identity to the formula at init and run at rung 5 only.

With `noise` (push + own noise) h also emits log_sigma (0 at init, sigma = 1), clamped to
[log 1e-3, log 1e3]: a Gaussian on log(1 + P) whose sigma is h's leftover error when the count
is known. For `affine`/`mlp` it is the second output of the same map; for `formula_affine` it is
gamma_assay + delta * f (the same per-assay-plus-slope form as mu; gamma, delta 0 at init).

Losses (per bin, mean over the bins of `mask`; y = the true P, never logged by the caller):
- push, true X':   (mu - log1p y)^2
- noise, true X':  Gaussian NLL of log1p y
- push, f's NB:    sum_k w_k (mu_k - log1p y)^2                       (design: the expected MSE)
- noise, f's NB:   rule "expected" sum_k w_k NLL_k (PI decision (iii), the default) or
                   rule "mixture"  -log sum_k w_k N(log1p y; mu_k, sigma_k) (behind the flag)

Import convention (plan): tools/t118 and tools/t125 on sys.path, then `from hmap import model`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[2]
for _p in (_TOOLS / "t118", _TOOLS / "t125"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import encoding  # noqa: E402
from ladder import model as ladder_model  # noqa: E402

HIDDEN = ladder_model.HIDDEN                       # 64, the t118 g's width
VARIANTS = ("affine", "mlp", "formula_affine", "formula_mlp")
RUNGS = (1, 5)
N_IN = {1: 9, 5: 19}                               # pinned in t125-P2a (FEATURES_R1 / FEATURES_R5)
N_ASSAY = len(encoding.ASSAYS)                     # 7
ASSAY_COLS = slice(2, 2 + N_ASSAY)                 # the one-hot, columns 2..8 at both rungs
HAS_CONTROL_COL = 18                               # rung 5's flag column
LOG_SIGMA_MIN = math.log(ladder_model.SIGMA_MIN)   # log 1e-3
LOG_SIGMA_MAX = math.log(ladder_model.SIGMA_MAX)   # log 1e3
STD_EPS = encoding.STD_EPS                         # 1e-6
NOISE_RULES = ("expected", "mixture")
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def continuous_cols(rung: int, formula: bool = False) -> list[int]:
    """The columns the standardiser z-scores: every continuous feature, never the assay one-hot,
    never `has_control`, and never the formula column (the formula variants add it to mu as is,
    so it must stay log1p of the formula's P). `formula` is accepted for symmetry; the formula
    column is last and not in the list either way."""
    if rung == 1:
        return [0, 1]
    if rung == 5:
        return [0, 1] + list(range(ASSAY_COLS.stop, HAS_CONTROL_COL))
    raise ValueError(f"rung {rung!r} not in {RUNGS}")


class Standardiser:
    """Per-column z-score over the chosen continuous columns; other columns pass unchanged.

    Population std; a column with std < 1e-6 maps to 0 (as `ladder.encoding`). Fitted once in
    float64; `apply` takes numpy or torch of any leading shape [..., F] and returns the same kind.
    """

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.cols: list[int] = []

    def fit(self, feat, continuous_cols) -> "Standardiser":
        a = feat.detach().cpu().numpy() if isinstance(feat, torch.Tensor) else np.asarray(feat)
        a = a.reshape(-1, a.shape[-1]).astype(np.float64)
        F = a.shape[1]
        cols = sorted(int(c) for c in continuous_cols)
        if any(c < 0 or c >= F for c in cols):
            raise ValueError(f"continuous_cols {cols} out of range for {F} columns")
        self.mean = np.zeros(F)
        self.std = np.ones(F)
        self.cols = cols
        if cols:
            self.mean[cols] = a[:, cols].mean(axis=0)
            self.std[cols] = a[:, cols].std(axis=0)
        return self

    def _coeffs(self) -> tuple[np.ndarray, np.ndarray]:
        """out = feat * scale + shift; scale 1 / shift 0 off the continuous columns."""
        scale = np.ones_like(self.mean)
        shift = np.zeros_like(self.mean)
        for c in self.cols:
            if self.std[c] < STD_EPS:
                scale[c], shift[c] = 0.0, 0.0
            else:
                scale[c], shift[c] = 1.0 / self.std[c], -self.mean[c] / self.std[c]
        return scale, shift

    def apply(self, feat):
        if self.mean is None:
            raise RuntimeError("Standardiser not fitted")
        if feat.shape[-1] != self.mean.shape[0]:
            raise ValueError(f"feature width {feat.shape[-1]}, fitted on {self.mean.shape[0]}")
        scale, shift = self._coeffs()
        if isinstance(feat, torch.Tensor):
            dt = feat.dtype if feat.is_floating_point() else torch.float32
            sc = torch.as_tensor(scale, dtype=dt, device=feat.device)
            sh = torch.as_tensor(shift, dtype=dt, device=feat.device)
            return feat.to(dt) * sc + sh
        a = np.asarray(feat)
        dt = a.dtype if np.issubdtype(a.dtype, np.floating) else np.float32
        return (a.astype(np.float64) * scale + shift).astype(dt)

    def state(self) -> dict:
        return {"mean": [float(v) for v in self.mean], "std": [float(v) for v in self.std],
                "continuous_cols": list(self.cols)}

    @classmethod
    def from_state(cls, state: dict) -> "Standardiser":
        s = cls()
        s.mean = np.asarray(state["mean"], dtype=np.float64)
        s.std = np.asarray(state["std"], dtype=np.float64)
        s.cols = [int(c) for c in state["continuous_cols"]]
        return s


def _mlp(n_in: int, n_out: int) -> torch.nn.Sequential:
    """MLP n_in -> 64 -> 64 -> n_out, ReLU; last layer weight 0, bias 0 (as the t118 G)."""
    net = torch.nn.Sequential(
        torch.nn.Linear(n_in, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, n_out))
    with torch.no_grad():
        net[-1].weight.zero_()
        net[-1].bias.zero_()
    return net


class HModel(torch.nn.Module):
    """One learned h. forward(feat [..., F]) -> (mu [...], log_sigma [...] | None)."""

    def __init__(self, variant: str, rung: int, noise: bool, n_in: int):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant {variant!r} not in {VARIANTS}")
        if rung not in RUNGS:
            raise ValueError(f"rung {rung!r} not in {RUNGS}")
        if n_in != N_IN[rung]:
            raise ValueError(f"rung {rung} takes {N_IN[rung]} features, got n_in {n_in}")
        self.formula = variant.startswith("formula")
        if self.formula and rung != 5:
            raise ValueError(f"{variant} runs at rung 5 only (the formula needs the rung-5 inputs)")
        self.variant, self.rung, self.noise, self.n_in = variant, int(rung), bool(noise), int(n_in)
        self.n_feat = n_in + 1 if self.formula else n_in
        n_out = 2 if self.noise else 1
        if variant == "affine":
            self.lin = torch.nn.Linear(self.n_feat, n_out)
            with torch.no_grad():
                self.lin.weight.zero_()
                self.lin.bias.zero_()
        elif variant in ("mlp", "formula_mlp"):
            self.net = _mlp(self.n_feat, n_out)
        else:                                       # formula_affine
            self.alpha = torch.nn.Parameter(torch.zeros(N_ASSAY))
            self.beta = torch.nn.Parameter(torch.zeros(1))
            if self.noise:
                self.gamma = torch.nn.Parameter(torch.zeros(N_ASSAY))
                self.delta = torch.nn.Parameter(torch.zeros(1))

    def forward(self, feat: torch.Tensor):
        if feat.shape[-1] != self.n_feat:
            raise ValueError(f"{self.variant} r{self.rung} takes {self.n_feat} columns, "
                             f"got {feat.shape[-1]}")
        if self.variant == "formula_affine":
            f = feat[..., -1]
            onehot = feat[..., ASSAY_COLS]
            mu = f + onehot @ self.alpha + self.beta * f
            ls = onehot @ self.gamma + self.delta * f if self.noise else None
        else:
            out = self.lin(feat) if self.variant == "affine" else self.net(feat)
            mu = out[..., 0]
            if self.formula:
                mu = feat[..., -1] + mu
            ls = out[..., 1] if self.noise else None
        if ls is not None:
            ls = ls.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return mu, ls


def P_hat(mu: torch.Tensor) -> torch.Tensor:
    """The P-scale point of mu = log(1 + P): expm1(mu), floored at 0."""
    return torch.expm1(mu).clamp(min=0)


def _mean(per_bin: torch.Tensor, mask) -> torch.Tensor:
    """Mean over the bins of mask (all bins when None); NaN in a masked-out bin cannot leak."""
    if mask is None:
        return per_bin.mean()
    mask = torch.as_tensor(mask, dtype=torch.bool, device=per_bin.device)
    kept = torch.where(mask, per_bin, torch.zeros_like(per_bin))
    return kept.sum() / mask.sum().clamp(min=1)


def _gauss_nll(t: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    z = (t - mu) * torch.exp(-log_sigma)
    return log_sigma + _HALF_LOG_2PI + 0.5 * z * z


def _target(y: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.as_tensor(y, device=like.device).to(like.dtype))


def loss_push_true(mu, y, mask=None) -> torch.Tensor:
    """Push only on the true X': MSE on log(1 + P)."""
    return _mean((mu - _target(y, mu)) ** 2, mask)


def loss_noise_true(mu, log_sigma, y, mask=None) -> torch.Tensor:
    """Own noise on the true X': Gaussian NLL of log(1 + P)."""
    return _mean(_gauss_nll(_target(y, mu), mu, log_sigma), mask)


def loss_push_fnb(mu, w, y, mask=None) -> torch.Tensor:
    """Push only on f's NB: sum_k w_k (mu_k - log1p y)^2, mu and w [N, K], y [N]."""
    t = _target(y, mu)[..., None]
    w = torch.as_tensor(w, device=mu.device).to(mu.dtype)
    return _mean((w * (mu - t) ** 2).sum(-1), mask)


def loss_noise_fnb(mu, log_sigma, w, y, rule: str = "expected", mask=None) -> torch.Tensor:
    """Own noise on f's NB. rule "expected" (default, PI decision (iii)): sum_k w_k NLL_k.
    rule "mixture": -log sum_k w_k N(log1p y; mu_k, sigma_k)."""
    if rule not in NOISE_RULES:
        raise ValueError(f"rule {rule!r} not in {NOISE_RULES}")
    t = _target(y, mu)[..., None]
    w = torch.as_tensor(w, device=mu.device).to(mu.dtype)
    nll = _gauss_nll(t, mu, log_sigma)
    if rule == "expected":
        return _mean((w * nll).sum(-1), mask)
    pos = w > 0
    log_w = torch.where(pos, torch.log(torch.where(pos, w, torch.ones_like(w))),
                        torch.full_like(w, -math.inf))
    row_ok = pos.any(-1, keepdim=True)              # a zero-weight (padded) row stays finite
    log_w = torch.where(row_ok, log_w, torch.zeros_like(w))
    return _mean(-torch.logsumexp(log_w - nll, dim=-1), mask)


def save_ckpt(path, h: HModel, standardiser: Standardiser, config: dict, step: int,
              val: float) -> None:
    """Write ckpt keys h, std, config, step, val_loss. config gains variant/rung/noise/n_in from h
    when absent; a present value that disagrees with h is an error."""
    cfg = dict(config)
    for key, val_h in (("variant", h.variant), ("rung", h.rung), ("noise", int(h.noise)),
                       ("n_in", h.n_in)):
        if key in cfg and cfg[key] != val_h:          # True == 1, so noise may be bool or 0/1
            raise ValueError(f"config {key}={cfg[key]!r} disagrees with h's {val_h!r}")
        cfg.setdefault(key, val_h)
    torch.save({"h": {k: v.detach().cpu() for k, v in h.state_dict().items()},
                "std": standardiser.state(), "config": cfg, "step": int(step),
                "val_loss": float(val)}, Path(path))


def load_ckpt(path, device="cpu"):
    """-> (HModel, Standardiser, config, step, val_loss)."""
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    cfg = ck["config"]
    rung = int(cfg["rung"])
    h = HModel(cfg["variant"], rung, bool(cfg["noise"]), int(cfg.get("n_in", N_IN[rung])))
    h.load_state_dict(ck["h"])
    h.to(device)
    return h, Standardiser.from_state(ck["std"]), cfg, int(ck["step"]), float(ck["val_loss"])
