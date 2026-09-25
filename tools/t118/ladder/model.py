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
pair i are those of pair perm[i], exactly as in training; any other (src, tgt) gets its own.
"""
from __future__ import annotations

import hashlib
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import nbinom, norm

from ladder import base, data, encoding

HIDDEN = 64
P_FLOOR = 1e-3
MU_MIN, MU_MAX = 1e-4, 1e6
N_MIN, N_MAX = 1e-3, 1e4
SIGMA_MIN, SIGMA_MAX = 1e-3, 1e3
CHUNK = 65_536
BATCH_CHUNKS = 8
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
    """Per-bin NB negative log-likelihood of count y, mean exp(loc), size exp(disp)."""
    log_mu = loc.clamp(math.log(MU_MIN), math.log(MU_MAX))
    log_n = disp.clamp(math.log(N_MIN), math.log(N_MAX))
    n = log_n.exp()
    log_n_mu = torch.logaddexp(log_n, log_mu)          # log(n + mu)
    ll = (torch.lgamma(y + n) - torch.lgamma(n) - torch.lgamma(y + 1.0)
          + n * (log_n - log_n_mu) + y * (log_mu - log_n_mu))
    return -ll


def lognormal_nll(loc: torch.Tensor, disp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-bin log-normal negative log-likelihood of y floored at 1e-3 (the floor is the loss's)."""
    t = torch.log(y.clamp(min=P_FLOOR))
    log_s = disp.clamp(math.log(SIGMA_MIN), math.log(SIGMA_MAX))
    z = (t - loc) / log_s.exp()
    return t + log_s + _HALF_LOG_2PI + 0.5 * z * z


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


class Ladder(torch.nn.Module):
    """g + one f-form: `forward(cov_pair [B, 40], x [B, L + 2c]) -> (loc [B, L], disp [B, L])`."""

    def __init__(self, rung: str, space: str, stats: dict, n_cov: int = encoding.N_COV):
        super().__init__()
        self.rung, self.space, self.n_cov = rung, space, int(n_cov)
        self.form = base.load_form(rung, space, stats)
        self.context = int(self.form.context)
        self.n_theta = int(self.form.n_theta)
        self.g = G(2 * self.n_cov, self.n_theta, self.form.init_theta())

    def theta(self, cov_pair: torch.Tensor) -> torch.Tensor:
        return self.g(cov_pair)

    def forward(self, cov_pair: torch.Tensor, x: torch.Tensor):
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
    `Corpus.window` pads during training).
    """
    c = ladder.context
    n = int(np.asarray(X).shape[0])
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


def md5_path(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def ids_cov_map(train_pairs: list[dict], perm) -> dict:
    """(src, tgt) of trained pair i -> (src, tgt) of pair perm[i]: the ids twin's covariate labels."""
    keys = [(p["source_pid"], p["target_pid"]) for p in train_pairs]
    return {keys[i]: keys[int(perm[i])] for i in range(len(keys))}


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
        self._cov_map = (ids_cov_map(self.train_pairs, config["ids_perm"])
                         if self.model == "ids" else {})

    def cov_pids(self, cov_src_pid: str, cov_tgt_pid: str) -> tuple[str, str]:
        """The pids whose covariates g actually receives (differs only for the ids twin)."""
        return self._cov_map.get((cov_src_pid, cov_tgt_pid), (cov_src_pid, cov_tgt_pid))

    def _theta_t(self, cov_src_pid: str, cov_tgt_pid: str) -> torch.Tensor:
        s, t = self.cov_pids(cov_src_pid, cov_tgt_pid)
        cov = torch.from_numpy(self.encoder.encode_pair(s, t)).to(self.device).reshape(1, -1)
        with torch.no_grad():
            return self.ladder.theta(cov)[0]

    def theta(self, cov_src_pid: str, cov_tgt_pid: str) -> np.ndarray:
        return self._theta_t(cov_src_pid, cov_tgt_pid).float().cpu().numpy()

    def describe(self, cov_src_pid: str, cov_tgt_pid: str) -> dict:
        return self.ladder.form.describe(self._theta_t(cov_src_pid, cov_tgt_pid).cpu())

    def predict(self, x_src_pid: str, cov_src_pid: str, cov_tgt_pid: str, chrom: str):
        """(loc, disp) float32[n] over the whole chromosome of `x_src_pid`."""
        X = self.corpus.get(x_src_pid, self.space, chrom)
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
