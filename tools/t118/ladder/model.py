"""t118 ladder — g, the Ladder (g + one f-form), the NB / log-normal losses, and the Predictor.

**g.** An MLP 40 -> 64 -> 64 -> n_theta with ReLU. Its input is concat(encode(C), encode(C')) —
never C' - C. The last layer's weight is zero and its bias is `form.init_theta()`, so every rung
starts at noSolution (theta = the identity map for every input).

**Loss** (per bin, mean over bins).
  counts  NB NLL on the raw count y: mu = exp(loc) clamped to [1e-4, 1e6], n = exp(disp) clamped
          to [1e-3, 1e4]; `torch.lgamma` for the normaliser.
  pval    log-normal NLL of y on t = log(max(y, 1e-3)) — the 1e-3 floor lives in the loss only:
          t + log sigma + log(2 pi)/2 + (t - loc)^2 / (2 sigma^2), sigma = exp(disp) clamped to
          [1e-3, 1e3] (the clamp keeps sigma from collapsing on the many floored bins where the
          source and the target are both 0).

**Predictor** (`load_run`). Rebuilds the Ladder from `ckpt.pt` and predicts whole chromosomes in
chunks of 65 536 bins with the form's context halo. The covariate pids may differ from the source
pid — that is how shuffle and swap are made. For the labels-as-ids twin, the covariates of trained
pair i are those of pair perm[i], exactly as in training; any other (src, tgt) gets its own. The
no-covariates twin is one average map: its Predictor uses ONE theta for every query — the mean of
g's theta over the run's training pairs — so g is never asked about a (C, C') it never saw
(foreman's ruling, 2026-09-25).

**Row 2** (forms with `per_bin = True`, plan/T118_ROW2_SPEC.md §2-§3). g is `GBin`: the same MLP
with one more input, the bin's own transformed value x_i, applied at every input position, so theta
is `[B, W, n_theta]`. For fixed (C, C') theta_i depends on x_i alone, so prediction evaluates g once
per distinct x value of the chromosome and gathers (`theta_levels`, `predict_chrom_bins`) — exact,
not an approximation. The row-2 no-covariates twin is one averaged map that still reads x:
theta_bar(x) = mean over the training pairs j of g(x, C_j, C'_j).

**Shuffled-bin twin** (model "xshuf", row 2, real covariates; `ladder/xshuf.py`). g reads x from
another bin of the same chromosome, f reads the true x: `Ladder.forward(cov, x, x_g)`. At
prediction `predict_chrom_bins(..., x_g_rng=xshuf.pred_rng(seed, chrom))` permutes the chromosome's
bins once for g (pads read themselves); the distinct-value gather is unchanged and still exact.
"""
from __future__ import annotations

import hashlib
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import nbinom, norm

from ladder import base, data, encoding, xshuf

HIDDEN = 64
P_FLOOR = 1e-3
MU_MIN, MU_MAX = 1e-4, 1e6
N_MIN, N_MAX = 1e-3, 1e4
SIGMA_MIN, SIGMA_MAX = 1e-3, 1e3
CHUNK = 65_536
BATCH_CHUNKS = 8
#: theta_levels evaluates g on at most this many (pair, level) rows at once
LEVEL_ROWS = 1 << 18
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def pick_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device {device!r} not in (auto, cpu, cuda)")
    return torch.device(device)


def transform_x(X: np.ndarray, space: str) -> np.ndarray:
    """f's input: counts log1p(X), pval log(max(X, 1e-3)); float32."""
    X = np.asarray(X, dtype=np.float64)
    if space == "counts":
        return np.log1p(X).astype(np.float32)
    if space == "pval":
        return np.log(np.maximum(X, P_FLOOR)).astype(np.float32)
    raise ValueError(f"space {space!r}")


def nb_nll(loc: torch.Tensor, disp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-bin NB negative log-likelihood of count y, mean exp(loc), size exp(disp).

    Computed in float64 and returned in loc's dtype: in float32 the lgamma differences lose
    ~1e-3 nats per bin at large n, a systematic error at small mu.
    """
    out_dtype = loc.dtype
    loc, disp, y = loc.double(), disp.double(), y.double()
    log_mu = loc.clamp(math.log(MU_MIN), math.log(MU_MAX))
    log_n = disp.clamp(math.log(N_MIN), math.log(N_MAX))
    n = log_n.exp()
    log_n_mu = torch.logaddexp(log_n, log_mu)          # log(n + mu)
    ll = (torch.lgamma(y + n) - torch.lgamma(n) - torch.lgamma(y + 1.0)
          + n * (log_n - log_n_mu) + y * (log_mu - log_n_mu))
    return (-ll).to(out_dtype)


def lognormal_nll(loc: torch.Tensor, disp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-bin log-normal negative log-likelihood of y floored at 1e-3 (the floor is the loss's).

    Computed in float64 and returned in loc's dtype, as `nb_nll`.
    """
    out_dtype = loc.dtype
    loc, disp, y = loc.double(), disp.double(), y.double()
    t = torch.log(y.clamp(min=P_FLOOR))
    log_s = disp.clamp(math.log(SIGMA_MIN), math.log(SIGMA_MAX))
    z = (t - loc) / log_s.exp()
    return (t + log_s + _HALF_LOG_2PI + 0.5 * z * z).to(out_dtype)


class G(torch.nn.Module):
    """MLP n_in -> 64 -> 64 -> n_theta, ReLU; last layer weight 0, bias = init_theta."""

    def __init__(self, n_in: int, n_theta: int, init_theta: torch.Tensor):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN, n_theta))
        last = self.net[-1]
        init_theta = torch.as_tensor(init_theta, dtype=torch.float32).reshape(-1)
        if init_theta.numel() != n_theta:
            raise ValueError(f"init_theta has {init_theta.numel()} entries, n_theta {n_theta}")
        with torch.no_grad():
            last.weight.zero_()
            last.bias.copy_(init_theta)

    def forward(self, cov_pair: torch.Tensor) -> torch.Tensor:
        return self.net(cov_pair)


class GBin(torch.nn.Module):
    """Row-2 g: MLP (1 + n_in) -> 64 -> 64 -> n_theta, ReLU, input [x_i, C, C'] at every position;
    last layer weight 0, bias = init_theta (theta_i = init_theta for every x_i at initialisation).

    The first layer is computed as `x * w[:, 0] + (cov @ w[:, 1:].T + b)`, so the [B, W, 1 + n_in]
    input is never built.
    """

    def __init__(self, n_in: int, n_theta: int, init_theta: torch.Tensor):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1 + n_in, HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN, n_theta))
        last = self.net[-1]
        init_theta = torch.as_tensor(init_theta, dtype=torch.float32).reshape(-1)
        if init_theta.numel() != n_theta:
            raise ValueError(f"init_theta has {init_theta.numel()} entries, n_theta {n_theta}")
        with torch.no_grad():
            last.weight.zero_()
            last.bias.copy_(init_theta)

    def forward(self, cov_pair: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """cov_pair [B, n_in], x [B, W] -> theta [B, W, n_theta]."""
        if cov_pair.dim() != 2 or x.dim() != 2 or cov_pair.shape[0] != x.shape[0]:
            raise ValueError(f"GBin: cov_pair {tuple(cov_pair.shape)} must be [B, n_in] and x "
                             f"{tuple(x.shape)} must be [B, W]")
        first = self.net[0]
        w, b = first.weight, first.bias
        c = torch.nn.functional.linear(cov_pair, w[:, 1:], b)            # [B, 64]
        h = x.to(w.dtype)[:, :, None] * w[:, 0] + c[:, None, :]          # [B, W, 64]
        for layer in list(self.net)[1:]:
            h = layer(h)
        return h

    def levels(self, cov_pair: torch.Tensor, x_values: torch.Tensor) -> torch.Tensor:
        """cov_pair [k, n_in], x_values [U] -> theta [k, U, n_theta]: g at every (pair, level)."""
        x_values = x_values.reshape(1, -1).to(cov_pair.dtype)
        return self(cov_pair, x_values.expand(cov_pair.shape[0], -1))


def _mlp(n_in: int, n_out: int, bias: torch.Tensor) -> torch.nn.Sequential:
    """MLP n_in -> 64 -> 64 -> n_out, ReLU; last layer weight 0, bias = `bias` (as `G`)."""
    net = torch.nn.Sequential(
        torch.nn.Linear(n_in, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, n_out))
    with torch.no_grad():
        net[-1].weight.zero_()
        net[-1].bias.copy_(torch.as_tensor(bias, dtype=torch.float32).reshape(-1))
    return net


class GEncDec(torch.nn.Module):
    """t126 encoder-decoder g for rung A's f (forms AED, AED1; plan/T126_ENCDEC_A.md).

    `phi`: ONE MLP n_cov -> 64 -> 64 -> (alpha, log beta) (or alpha alone when `beta_learned` is
    False, beta = 1), applied to the source covariates c and the target covariates c' separately.
    `psi`: MLP n_cov -> 64 -> 64 -> 1 on c' only, the NB log size d. Returns rung A's theta:
        b = beta(c') / beta(c),  a = alpha(c') - b * alpha(c),  d = psi(c').
    Input `cov_pair` [B, 2 n_cov] = concat(encode(c), encode(c')), as `G`. Last layers have weight 0;
    phi's bias is 0 (alpha = 0, beta = 1) and psi's bias is init_theta's d, so theta starts at
    `init_theta` = (0, 1, d0) for every input, as every rung does.
    """

    def __init__(self, n_cov: int, init_theta: torch.Tensor, beta_learned: bool = True):
        super().__init__()
        init_theta = torch.as_tensor(init_theta, dtype=torch.float32).reshape(-1)
        if init_theta.numel() != 3 or float(init_theta[0]) != 0.0 or float(init_theta[1]) != 1.0:
            raise ValueError(f"GEncDec needs rung A's init_theta (0, 1, d0), got {init_theta.tolist()}")
        self.n_cov = int(n_cov)
        self.beta_learned = bool(beta_learned)
        self.phi = _mlp(self.n_cov, 2 if self.beta_learned else 1, torch.zeros(2 if beta_learned else 1))
        self.psi = _mlp(self.n_cov, 1, init_theta[2:3])

    def latent(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """c [B, n_cov] -> (alpha [B], log beta [B]); log beta = 0 when beta is fixed to 1."""
        out = self.phi(c)
        alpha = out[:, 0]
        log_beta = out[:, 1] if self.beta_learned else torch.zeros_like(alpha)
        return alpha, log_beta

    def forward(self, cov_pair: torch.Tensor) -> torch.Tensor:
        if cov_pair.dim() != 2 or cov_pair.shape[1] != 2 * self.n_cov:
            raise ValueError(f"GEncDec: cov_pair {tuple(cov_pair.shape)} must be [B, {2 * self.n_cov}]")
        c, ct = cov_pair[:, :self.n_cov], cov_pair[:, self.n_cov:]
        alpha, log_beta = self.latent(c)
        alpha_t, log_beta_t = self.latent(ct)
        b = torch.ones_like(alpha) if not self.beta_learned else torch.exp(log_beta_t - log_beta)
        a = alpha_t - b * alpha
        d = self.psi(ct)[:, 0]
        return torch.stack([a, b, d], dim=1)


class GEncDecD(torch.nn.Module):
    """t126 encoder-decoder g for rung DED (`fforms/form_ded.py`; plan/T126_ENCDEC_A.md "Rung D").

    `g_enc`: MLP n_cov -> 64 -> 64 -> 128 on the source covariates c: the encoder's FiLM
    (gamma[2][32], beta[2][32]). `g_dec`: a separate MLP of the same shape on c' only: the
    decoder's FiLM. Last layers weight 0, bias gamma 1 / beta 0. `forward(cov_pair)` returns the
    form's D-layout theta [B, 256] (`FormDED.join_theta`), so the Ladder, the Predictor and the
    no-covariates twin's mean theta work as for any row-1 rung.
    """

    def __init__(self, n_cov: int, n_side: int):
        super().__init__()
        self.n_cov = int(n_cov)
        half = n_side // 2
        init = torch.cat([torch.ones(half), torch.zeros(half)])
        self.g_enc = _mlp(self.n_cov, n_side, init)
        self.g_dec = _mlp(self.n_cov, n_side, init)

    def forward(self, cov_pair: torch.Tensor) -> torch.Tensor:
        from ladder.fforms.form_ded import FormDED
        if cov_pair.dim() != 2 or cov_pair.shape[1] != 2 * self.n_cov:
            raise ValueError(f"GEncDecD: cov_pair {tuple(cov_pair.shape)} must be [B, {2 * self.n_cov}]")
        c, ct = cov_pair[:, :self.n_cov], cov_pair[:, self.n_cov:]
        return FormDED.join_theta(self.g_enc(c), self.g_dec(ct))


class GSym(torch.nn.Module):
    """t128 symmetric g for rungs DS / DSBOX (`fforms/form_ds.py`; plan/T126_ENCDEC_D2.md).

    ONE MLP 2 n_cov -> 64 -> 64 -> 256 (`mlp`; last layer weight 0, bias gamma 1 / beta 0) gives
    old D's FiLM for a pair (a, b) of covariate vectors. A learned clean covariate vector C_clean
    sits in the middle: its 13 processing knobs are learned per assay (`cclean` [7, 13]); its assay
    entries are copied from the source c. With cc = C_clean of c,
        theta = cat(mlp(cat(c, cc)), mlp(cat(cc, c')))   [B, 512]
    the encoder's FiLM then the decoder's. Form A (`box` False): the knobs are `cclean` itself.
    Form D (`box` True): `cclean` is the pre-sigmoid u, knobs = lo + (hi - lo) * sigmoid(u), with
    the box [lo, hi] set by `set_box`. The init values are the training script's job
    (`set_cclean_init`; PI 2026-10-21: zeros for form A, the base's covariates for form D).
    """

    def __init__(self, n_cov: int, init_theta: torch.Tensor, box: bool):
        super().__init__()
        from ladder.fforms.form_ds import ASSAY_SLICE, FILM, N_ASSAY, N_KNOB
        init_theta = torch.as_tensor(init_theta, dtype=torch.float32).reshape(-1)
        if init_theta.numel() != 2 * FILM:
            raise ValueError(f"GSym: init_theta has {init_theta.numel()} entries, want {2 * FILM}")
        if int(n_cov) != ASSAY_SLICE.stop:
            raise ValueError(f"GSym: n_cov {n_cov} != {ASSAY_SLICE.stop} (13 knobs + 7 assays)")
        self.n_cov = int(n_cov)
        self.box = bool(box)
        self.mlp = _mlp(2 * self.n_cov, FILM, init_theta[:FILM])
        self.cclean = torch.nn.Parameter(torch.zeros(N_ASSAY, N_KNOB))
        self.register_buffer("box_lo", torch.zeros(N_ASSAY, N_KNOB))
        self.register_buffer("box_hi", torch.zeros(N_ASSAY, N_KNOB))

    @staticmethod
    def assay_index(c: torch.Tensor) -> torch.Tensor:
        from ladder.fforms.form_ds import ASSAY_SLICE
        return c[:, ASSAY_SLICE].argmax(dim=1)

    def cclean_knobs(self) -> torch.Tensor:
        """[7, 13]: the decoded knobs of C_clean, one row per assay."""
        if not self.box:
            return self.cclean
        return self.box_lo + (self.box_hi - self.box_lo) * torch.sigmoid(self.cclean)

    def cclean_of(self, c: torch.Tensor) -> torch.Tensor:
        """c [B, 20] -> C_clean [B, 20]: the knobs of c's assay, then c's own assay entries."""
        from ladder.fforms.form_ds import ASSAY_SLICE
        knobs = self.cclean_knobs()[self.assay_index(c)].to(c.dtype)
        return torch.cat([knobs, c[:, ASSAY_SLICE]], dim=1)

    def film(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a, b [B, 20] -> old D's FiLM [B, 256]."""
        return self.mlp(torch.cat([a, b], dim=1))

    def forward(self, cov_pair: torch.Tensor) -> torch.Tensor:
        if cov_pair.dim() != 2 or cov_pair.shape[1] != 2 * self.n_cov:
            raise ValueError(f"GSym: cov_pair {tuple(cov_pair.shape)} must be [B, {2 * self.n_cov}]")
        c, ct = cov_pair[:, :self.n_cov], cov_pair[:, self.n_cov:]
        cc = self.cclean_of(c)
        return torch.cat([self.film(c, cc), self.film(cc, ct)], dim=1)

    @torch.no_grad()
    def set_box(self, lo: torch.Tensor, hi: torch.Tensor) -> None:
        """Copy the box [lo, hi] ([7, 13] each). Form D only; a no-op with a warning for form A."""
        if not self.box:
            warnings.warn("GSym.set_box: form A (free C_clean) has no box; ignored")
            return
        lo = torch.as_tensor(lo, dtype=self.box_lo.dtype).reshape(self.box_lo.shape)
        hi = torch.as_tensor(hi, dtype=self.box_hi.dtype).reshape(self.box_hi.shape)
        if bool((hi < lo).any()):
            raise ValueError("GSym.set_box: hi < lo somewhere")
        self.box_lo.copy_(lo)
        self.box_hi.copy_(hi)

    @torch.no_grad()
    def set_cclean_init(self, values: torch.Tensor) -> None:
        """Initialise C_clean's knobs at `values` [7, 13]. Form A copies them; form D stores
        clamp(logit((values - lo) / (hi - lo)), -4, 4), 0 where hi == lo (call `set_box` first)."""
        v = torch.as_tensor(values, dtype=self.cclean.dtype).reshape(self.cclean.shape)
        if not self.box:
            self.cclean.copy_(v)
            return
        width = self.box_hi - self.box_lo
        flat = width <= 0
        frac = (v - self.box_lo) / torch.where(flat, torch.ones_like(width), width)
        u = torch.logit(frac.clamp(1e-12, 1.0 - 1e-12)).clamp(-4.0, 4.0)
        self.cclean.copy_(torch.where(flat, torch.zeros_like(u), u))

    @torch.no_grad()
    def cclean_table(self) -> dict:
        """{assay name: [13 decoded knob values]}."""
        k = self.cclean_knobs().detach().double().cpu()
        return {a: k[i].tolist() for i, a in enumerate(encoding.ASSAYS)}


class Ladder(torch.nn.Module):
    """g + one f-form: `forward(cov_pair [B, 40], x [B, L + 2c]) -> (loc [B, L], disp [B, L])`.

    Row 2 (`per_bin`, the form's flag): g is `GBin` and reads x as well; the signature is the same.
    `x_g` (row 2 only, shaped like x): the x g reads instead of x — the shuffled-bin twin.
    t126: a form whose `g_kind` is "encdec" / "encdec_beta1" (AED, AED1) gets `GEncDec`,
    "encdec_d" (DED) gets `GEncDecD`; a form without `g_kind` gets `G`, as before.
    """

    def __init__(self, rung: str, space: str, stats: dict, n_cov: int = encoding.N_COV):
        super().__init__()
        self.rung, self.space, self.n_cov = rung, space, int(n_cov)
        self.form = base.load_form(rung, space, stats)
        self.context = int(self.form.context)
        self.n_theta = int(self.form.n_theta)
        self.per_bin = bool(self.form.per_bin)
        g_kind = getattr(self.form, "g_kind", None)
        if self.per_bin:
            self.g = GBin(2 * self.n_cov, self.n_theta, self.form.init_theta())
        elif g_kind in ("encdec", "encdec_beta1"):
            self.g = GEncDec(self.n_cov, self.form.init_theta(), beta_learned=g_kind == "encdec")
        elif g_kind == "encdec_d":
            from ladder.fforms.form_ded import N_SIDE
            self.g = GEncDecD(self.n_cov, N_SIDE)
        elif g_kind == "sym_film":
            self.g = GSym(self.n_cov, self.form.init_theta(),
                          box=bool(getattr(self.form, "cclean_box", False)))
        elif g_kind is not None:
            raise ValueError(f"rung {rung}: unknown g_kind {g_kind!r}")
        else:
            self.g = G(2 * self.n_cov, self.n_theta, self.form.init_theta())

    def theta(self, cov_pair: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """Row 1: [B, n_theta] (x ignored). Row 2: x [B, W] required -> [B, W, n_theta]."""
        if not self.per_bin:
            return self.g(cov_pair)
        if x is None:
            raise ValueError(f"rung {self.rung}: a per-bin g needs x")
        return self.g(cov_pair, x)

    def forward(self, cov_pair: torch.Tensor, x: torch.Tensor, x_g: torch.Tensor | None = None):
        if self.per_bin:
            if x_g is not None and tuple(x_g.shape) != tuple(x.shape):
                raise ValueError(f"x_g {tuple(x_g.shape)} must be shaped like x {tuple(x.shape)}")
            return self.form(x, self.g(cov_pair, x if x_g is None else x_g))
        if x_g is not None:
            raise ValueError(f"rung {self.rung}: x_g needs a per-bin (row-2) form")
        return self.form(x, self.g(cov_pair))

    def nll_bins(self, loc, disp, y) -> torch.Tensor:
        return nb_nll(loc, disp, y) if self.space == "counts" else lognormal_nll(loc, disp, y)

    def nll(self, loc, disp, y, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mean per-bin NLL (over the bins where `mask` is True, if given)."""
        per = self.nll_bins(loc, disp, y)
        if mask is None:
            return per.mean()
        m = mask.to(per.dtype)
        return (per * m).sum() / m.sum().clamp(min=1.0)


@torch.no_grad()
def predict_chrom(ladder: Ladder, X: np.ndarray, theta: torch.Tensor, device,
                  chunk: int = CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """(loc, disp) float32[n] for a whole chromosome X of the source, one theta [n_theta].

    Chunks of `chunk` bins, each with `context` bins of halo; outside the chromosome X = 0 (as
    `Corpus.window` pads during training). A chromosome shorter than `chunk` is one chunk of its
    own length (not padded up to `chunk`: the synthetic test chromosomes are ~1000 bins).
    """
    c = ladder.context
    n = int(np.asarray(X).shape[0])
    chunk = max(1, min(int(chunk), n))
    n_chunks = max(1, -(-n // chunk))
    Xp = np.zeros(n_chunks * chunk + 2 * c, dtype=np.float64)
    Xp[c:c + n] = X
    xp = transform_x(Xp, ladder.space)
    theta = theta.reshape(1, -1).to(device=device, dtype=torch.float32)
    locs, disps = [], []
    for b0 in range(0, n_chunks, BATCH_CHUNKS):
        idx = range(b0, min(b0 + BATCH_CHUNKS, n_chunks))
        xb = np.stack([xp[i * chunk: i * chunk + chunk + 2 * c] for i in idx])
        xt = torch.from_numpy(xb).to(device)
        loc, disp = ladder.form(xt, theta.expand(xt.shape[0], -1))
        locs.append(loc.reshape(-1).float().cpu().numpy())
        disps.append(disp.reshape(-1).float().cpu().numpy())
    return (np.concatenate(locs)[:n].astype(np.float32),
            np.concatenate(disps)[:n].astype(np.float32))


@torch.no_grad()
def theta_levels(ladder: Ladder, x_values: torch.Tensor, cov: torch.Tensor,
                 average: bool = False) -> torch.Tensor:
    """Row 2: theta [U, n_theta] of g at the scalar levels `x_values` [U].

    `average=False`: `cov` must be [1, 40], one (C, C'). `average=True`: the mean over the k rows of
    `cov` [k, 40] of g(x, cov_j) — the no-covariates twin's map (spec §3). Evaluated in blocks of
    levels so k * block stays under LEVEL_ROWS.

    `average=True` takes g's last layer out of the mean: that layer is linear and nothing follows it,
    so mean_j(W h_j + b) = W mean_j(h_j) + b, with h_j g's last hidden layer for pair j. Exact up to
    float summation order; the last layer runs on U rows instead of k * U.
    """
    if not ladder.per_bin:
        raise ValueError(f"rung {ladder.rung}: theta_levels needs a per-bin form")
    cov = cov.reshape(-1, cov.shape[-1])
    k = int(cov.shape[0])
    if not average and k != 1:
        raise ValueError(f"theta_levels: {k} covariate rows without average=True")
    x_values = x_values.reshape(-1)
    step = max(1, LEVEL_ROWS // k)
    out = []
    if average:
        layers = list(ladder.g.net)
        last = layers[-1]
        if not isinstance(last, torch.nn.Linear):
            raise TypeError(f"g's last layer is {type(last).__name__}, not Linear: the mean over "
                            f"pairs cannot be taken before it")
        w, b = layers[0].weight, layers[0].bias
        c = torch.nn.functional.linear(cov, w[:, 1:], b)                 # [k, 64], as GBin.forward
        for u0 in range(0, max(1, x_values.shape[0]), step):
            xv = x_values[u0:u0 + step].to(cov.dtype).to(w.dtype)
            h = xv[None, :, None] * w[:, 0] + c[:, None, :]              # [k, u, 64]
            for layer in layers[1:-1]:
                h = layer(h)
            out.append(last(h.mean(0)))                                  # [u, n_theta]
        return torch.cat(out, 0)
    for u0 in range(0, max(1, x_values.shape[0]), step):
        t = ladder.g.levels(cov, x_values[u0:u0 + step])              # [k, u, n_theta]
        out.append(t[0])
    return torch.cat(out, 0)


@torch.no_grad()
def predict_chrom_bins(ladder: Ladder, X: np.ndarray, cov: torch.Tensor, device,
                       average: bool = False, chunk: int = CHUNK,
                       x_g_rng: np.random.Generator | None = None,
                       level_cache: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Row 2: (loc, disp) float32[n] for a whole chromosome X, theta per position from g.

    Same padding, chunks and halo as `predict_chrom`. theta at position j = `theta_levels` at the
    distinct values of the padded, transformed X, gathered back by the inverse index (exact: theta_j
    depends on x_j alone). `cov` [1, 40], or [k, 40] with `average=True` (the nocov twin).

    `x_g_rng` (the shuffled-bin twin): perm = `xshuf.draw_positions(x_g_rng, n, n, fixed=True)`;
    g at padded position q reads x at `src[q]`, `src[c + j] = c + perm[j]` on the bins and
    `src[q] = q` on the pads; f reads the true x. None: g reads the true x.

    `level_cache` (the nocov twin's validation): a dict owned by the caller for ONE fixed X. Empty,
    it is filled with this X's distinct values and inverse index; filled, they are reused instead of
    `np.unique` (the same arrays, so the output is bit-identical).
    """
    c = ladder.context
    n = int(np.asarray(X).shape[0])
    chunk = max(1, min(int(chunk), n))
    n_chunks = max(1, -(-n // chunk))
    Xp = np.zeros(n_chunks * chunk + 2 * c, dtype=np.float64)
    Xp[c:c + n] = X
    xp = transform_x(Xp, ladder.space)
    if level_cache:
        if level_cache["n_padded"] != xp.shape[0]:
            raise ValueError(f"level_cache holds {level_cache['n_padded']} padded bins, "
                             f"this X has {xp.shape[0]}")
        uniq, inv = level_cache["uniq"], level_cache["inv"]
    else:
        uniq, inv = np.unique(xp, return_inverse=True)
        if level_cache is not None:
            level_cache.update(n_padded=xp.shape[0], uniq=uniq,
                               inv=np.asarray(inv, dtype=np.int32 if uniq.size < 2**31 else np.int64))
    theta_u = theta_levels(ladder, torch.from_numpy(uniq).to(device),
                           cov.to(device=device, dtype=torch.float32), average)
    inv_t = torch.from_numpy(np.asarray(inv, dtype=np.int64).reshape(-1)).to(device)
    if x_g_rng is not None:
        perm = xshuf.draw_positions(x_g_rng, n, n, fixed=True)
        src = np.arange(xp.shape[0], dtype=np.int64)
        src[c:c + n] = c + perm
        inv_t = inv_t[torch.from_numpy(src).to(device)]
    offs = torch.arange(chunk + 2 * c, device=device)
    locs, disps = [], []
    for b0 in range(0, n_chunks, BATCH_CHUNKS):
        idx = range(b0, min(b0 + BATCH_CHUNKS, n_chunks))
        xb = np.stack([xp[i * chunk: i * chunk + chunk + 2 * c] for i in idx])
        xt = torch.from_numpy(xb).to(device)
        pos = torch.tensor([i * chunk for i in idx], device=device)[:, None] + offs
        loc, disp = ladder.form(xt, theta_u[inv_t[pos]])
        locs.append(loc.reshape(-1).float().cpu().numpy())
        disps.append(disp.reshape(-1).float().cpu().numpy())
    return (np.concatenate(locs)[:n].astype(np.float32),
            np.concatenate(disps)[:n].astype(np.float32))


def md5_path(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def ids_cov_map(train_pairs: list[dict], perm) -> dict:
    """(src, tgt) of trained pair i -> (src, tgt) of pair perm[i]: the ids twin's covariate labels."""
    keys = [(p["source_pid"], p["target_pid"]) for p in train_pairs]
    return {keys[i]: keys[int(perm[i])] for i in range(len(keys))}


@torch.no_grad()
def nocov_theta(ladder: Ladder, cov_train: torch.Tensor) -> torch.Tensor:
    """The no-covariates twin's single theta [n_theta]: g's theta averaged over the training pairs'
    (C, C') vectors `cov_train` [n_pairs, 40]."""
    return ladder.theta(cov_train).mean(0)


class Predictor:
    """A trained run, ready to predict. Built by `load_run`."""

    def __init__(self, ladder: Ladder, encoder: encoding.Encoder, config: dict, corpus,
                 device, step: int, val_nll: float):
        self.ladder = ladder.to(device).eval()
        self.encoder = encoder
        self.config = config
        self.corpus = corpus
        self.device = device
        self.step, self.val_nll = step, val_nll
        self.rung, self.space = config["rung"], config["space"]
        self.g_id, self.model, self.seed = config["g"], config["model"], int(config["seed"])
        self.run_name = config["run_name"]
        self.fit_pids = list(config["fit_pids"])
        self.train_pairs = list(config["train_pairs"])
        self.context = ladder.context
        self.n_theta = ladder.n_theta
        self.per_bin = ladder.per_bin
        self._cov_map = (ids_cov_map(self.train_pairs, config["ids_perm"])
                         if self.model == "ids" else {})
        self._nocov_theta = None
        self._nocov_cov = None
        if self.model == "nocov" and self.per_bin:
            self._nocov_cov = torch.from_numpy(np.stack([
                encoder.encode_pair(p["source_pid"], p["target_pid"])
                for p in self.train_pairs])).to(device)
        elif self.model == "nocov":
            cov = torch.from_numpy(np.stack([
                encoder.encode_pair(p["source_pid"], p["target_pid"]) for p in self.train_pairs]))
            self._nocov_theta = nocov_theta(self.ladder, cov.to(device))

    def cov_pids(self, cov_src_pid: str, cov_tgt_pid: str) -> tuple[str, str]:
        """The pids whose covariates g actually receives (differs only for the ids twin)."""
        return self._cov_map.get((cov_src_pid, cov_tgt_pid), (cov_src_pid, cov_tgt_pid))

    def _theta_t(self, cov_src_pid: str, cov_tgt_pid: str) -> torch.Tensor:
        if self._nocov_theta is not None:
            return self._nocov_theta
        s, t = self.cov_pids(cov_src_pid, cov_tgt_pid)
        cov = torch.from_numpy(self.encoder.encode_pair(s, t)).to(self.device).reshape(1, -1)
        with torch.no_grad():
            return self.ladder.theta(cov)[0]

    def _cov_t(self, cov_src_pid: str, cov_tgt_pid: str) -> tuple[torch.Tensor, bool]:
        """Row 2: (the covariate rows g receives, average?) — nocov: all training pairs, averaged."""
        if self._nocov_cov is not None:
            return self._nocov_cov, True
        s, t = self.cov_pids(cov_src_pid, cov_tgt_pid)
        return torch.from_numpy(self.encoder.encode_pair(s, t)).to(self.device).reshape(1, -1), False

    def _theta_levels_t(self, cov_src_pid: str, cov_tgt_pid: str) -> torch.Tensor:
        """Row 2: theta [12, n_theta] at the levels `stats["knots_x"]`."""
        cov, avg = self._cov_t(cov_src_pid, cov_tgt_pid)
        levels = torch.as_tensor(self.ladder.form.stats["knots_x"], dtype=torch.float32)
        return theta_levels(self.ladder, levels.to(self.device), cov, average=avg)

    def theta(self, cov_src_pid: str, cov_tgt_pid: str) -> np.ndarray:
        """Row 1: [n_theta]. Row 2: [12, n_theta] at the levels `stats["knots_x"]`."""
        if self.per_bin:
            return self._theta_levels_t(cov_src_pid, cov_tgt_pid).float().cpu().numpy()
        return self._theta_t(cov_src_pid, cov_tgt_pid).float().cpu().numpy()

    def describe(self, cov_src_pid: str, cov_tgt_pid: str) -> dict:
        """Row 1: `form.describe(theta)`. Row 2: `form.describe(theta [12, n_theta])` plus
        "levels_x" and "response" {"loc", "disp"}: the model on a flat window of width
        2*context + 1 whose bins all equal the level, centre bin."""
        if not self.per_bin:
            return self.ladder.form.describe(self._theta_t(cov_src_pid, cov_tgt_pid).cpu())
        theta = self._theta_levels_t(cov_src_pid, cov_tgt_pid)
        levels = torch.as_tensor(self.ladder.form.stats["knots_x"], dtype=torch.float32)
        w = 2 * self.context + 1
        x = levels.to(self.device)[:, None].expand(-1, w).contiguous()
        with torch.no_grad():
            loc, disp = self.ladder.form(x, theta[:, None, :].expand(-1, w, -1).contiguous())
        out = self.ladder.form.describe(theta.cpu())
        out["levels_x"] = [float(v) for v in self.ladder.form.stats["knots_x"]]
        out["response"] = {"loc": [float(v) for v in loc[:, 0].cpu()],
                           "disp": [float(v) for v in disp[:, 0].cpu()]}
        return out

    def predict(self, x_src_pid: str, cov_src_pid: str, cov_tgt_pid: str, chrom: str):
        """(loc, disp) float32[n] over the whole chromosome of `x_src_pid`."""
        X = self.corpus.get(x_src_pid, self.space, chrom)
        if self.per_bin:
            cov, avg = self._cov_t(cov_src_pid, cov_tgt_pid)
            rng = xshuf.pred_rng(self.seed, chrom) if self.model == "xshuf" else None
            return predict_chrom_bins(self.ladder, X, cov, self.device, average=avg, x_g_rng=rng)
        return predict_chrom(self.ladder, X, self._theta_t(cov_src_pid, cov_tgt_pid), self.device)

    def mean(self, loc: np.ndarray) -> np.ndarray:
        """counts: the NB mean mu = exp(loc) (clamped as in the loss); pval: exp(loc), the median."""
        loc = np.asarray(loc, dtype=np.float64)
        if self.space == "counts":
            return np.exp(np.clip(loc, math.log(MU_MIN), math.log(MU_MAX)))
        return np.exp(loc)

    def quantiles(self, loc: np.ndarray, disp: np.ndarray, q) -> np.ndarray:
        """Quantiles of the predictive distribution in the target's units; `q` scalar -> [n],
        `q` array -> [len(q), n]. NB via scipy.stats.nbinom.ppf; log-normal exp(loc + sigma z_q)."""
        loc = np.asarray(loc, dtype=np.float64)
        disp = np.asarray(disp, dtype=np.float64)
        qa = np.atleast_1d(np.asarray(q, dtype=np.float64))
        if self.space == "counts":
            mu = self.mean(loc)
            n = np.exp(np.clip(disp, math.log(N_MIN), math.log(N_MAX)))
            p = n / (n + mu)
            out = np.stack([nbinom.ppf(qi, n, p) for qi in qa])
        else:
            s = np.exp(np.clip(disp, math.log(SIGMA_MIN), math.log(SIGMA_MAX)))
            out = np.stack([np.exp(loc + s * norm.ppf(qi)) for qi in qa])
        return out[0] if np.ndim(q) == 0 else out


def load_run(run_dir, data_dir, covariates_tsv, manifest_tsv, device="auto") -> Predictor:
    """The Predictor of one finished run directory (reads `<run_dir>/ckpt.pt`).

    The covariate vectors come from the encoder stored in the checkpoint (what g was trained on);
    `covariates_tsv` / `manifest_tsv` are checked against the md5s in the run's config and a
    mismatch is a warning, not an error.
    """
    run_dir = Path(run_dir)
    dev = pick_device(device)
    ckpt = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    for path, key in ((covariates_tsv, "covariates_md5"), (manifest_tsv, "manifest_md5")):
        if path is not None and cfg.get(key) and md5_path(path) != cfg[key]:
            warnings.warn(f"{path}: md5 differs from the run's {key}; using the checkpoint's "
                          f"encoder")
    stats = {"knots_x": np.asarray(ckpt["stats"]["knots_x"], dtype=np.float64),
             "n0": float(ckpt["stats"]["n0"]), "sigma0": float(ckpt["stats"]["sigma0"])}
    ladder = Ladder(cfg["rung"], cfg["space"], stats, n_cov=int(cfg.get("n_cov", encoding.N_COV)))
    ladder.g.load_state_dict(ckpt["g"])
    ladder.form.load_state_dict(ckpt["form"])
    enc = encoding.Encoder.from_state(ckpt["encoder"])
    corpus = data.Corpus(data_dir)
    return Predictor(ladder, enc, cfg, corpus, dev, int(ckpt["step"]), float(ckpt["val_nll"]))
