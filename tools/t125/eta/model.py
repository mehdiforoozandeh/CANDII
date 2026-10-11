"""t125 part 1 — g + f with the depth offset, prediction, and the run loader (plan/ETA_OFFSET_AND_H_DESIGN.md).

**Offset.** f reads eta_in = log(1 + X) - log d (d = source depth in millions of reads,
`offset.transform_eta`) and emits eta_out; the target NB has log mu' = log d' + eta_out. f never
reads C; g reads (C, C') (row 2 also eta_in) and keeps depth as a covariate. With f at its
identity (eta_out = eta_in) the predicted mean is (1 + X) d'/d: "do nothing" = scale by d'/d.

**Reuse.** `ladder/model.py` is not edited. `EtaLadder` builds t118's `G` / `GBin` around the rung's
form (`load_form`, from `eta.pairs.FORM_MODULE`), so t118's `theta_levels` and `nocov_theta` take it
as they take a `Ladder`. Whole-chromosome prediction keeps t118's padding (X = 0 outside the
chromosome, so eta pad = -log d_src), chunks of 65 536 bins in batches of 8 and the context halo.
`EtaPredictor` subclasses t118's `Predictor` (theta, describe, mean, quantiles are its own) and
adds the per-pid log depths from `config["log_depth"]`. No ids twin: models are real and nocov.
"""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[2]
for _p in (_TOOLS / "t125", _TOOLS / "t118"):   # import convention: tools/t118 and tools/t125
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import base, data, encoding  # noqa: E402
from ladder import model as lmodel  # noqa: E402
from eta import offset  # noqa: E402
from eta import pairs as ep  # noqa: E402

SPACE = "counts"
MODELS = ("real", "nocov")


def load_form(rung: str, stats: dict) -> base.FForm:
    """`importlib.import_module(FORM_MODULE[rung]).build("counts", stats)`; KeyError for an unknown rung."""
    if rung not in ep.FORM_MODULE:
        raise KeyError(f"rung {rung!r} not in {tuple(ep.FORM_MODULE)}")
    form = importlib.import_module(ep.FORM_MODULE[rung]).build(SPACE, stats)
    if not isinstance(form, base.FForm):
        raise TypeError(f"{ep.FORM_MODULE[rung]}.build returned {type(form).__name__}, not an FForm")
    return form


class EtaLadder(torch.nn.Module):
    """g + one f-form over eta: `forward(cov_pair [B, 40], x_eta [B, L + 2c], log_d_tgt [B])
    -> (loc [B, L], disp [B, L])` with loc = log d' + eta_out."""

    def __init__(self, rung: str, stats: dict, n_cov: int = encoding.N_COV):
        super().__init__()
        self.rung, self.space, self.n_cov = rung, SPACE, int(n_cov)
        self.form = load_form(rung, stats)
        self.context = int(self.form.context)
        self.n_theta = int(self.form.n_theta)
        self.per_bin = bool(self.form.per_bin)
        cls = lmodel.GBin if self.per_bin else lmodel.G
        self.g = cls(2 * self.n_cov, self.n_theta, self.form.init_theta())

    def theta(self, cov_pair: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """Row 1: [B, n_theta] (x ignored). Row 2: x [B, W] required -> [B, W, n_theta]."""
        if not self.per_bin:
            return self.g(cov_pair)
        if x is None:
            raise ValueError(f"rung {self.rung}: a per-bin g needs x")
        return self.g(cov_pair, x)

    def form_out(self, cov_pair: torch.Tensor, x_eta: torch.Tensor):
        """(eta_out [B, L], disp [B, L]): f on eta_in, no offset."""
        return self.form(x_eta, self.theta(cov_pair, x_eta if self.per_bin else None))

    def forward(self, cov_pair: torch.Tensor, x_eta: torch.Tensor, log_d_tgt: torch.Tensor):
        log_d_tgt = torch.as_tensor(log_d_tgt).reshape(-1)
        if log_d_tgt.shape[0] != x_eta.shape[0]:
            raise ValueError(f"log_d_tgt has {log_d_tgt.shape[0]} entries, x_eta {x_eta.shape[0]} rows")
        loc, disp = self.form_out(cov_pair, x_eta)
        return loc + log_d_tgt.to(device=loc.device, dtype=loc.dtype)[:, None], disp

    def nll_bins(self, loc, disp, y) -> torch.Tensor:
        return lmodel.nb_nll(loc, disp, y)

    def nll(self, loc, disp, y, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mean per-bin NB NLL (over the bins where `mask` is True, if given)."""
        per = self.nll_bins(loc, disp, y)
        if mask is None:
            return per.mean()
        m = mask.to(per.dtype)
        return (per * m).sum() / m.sum().clamp(min=1.0)


def _padded_eta(X: np.ndarray, c: int, chunk: int, log_d_src: float):
    n = int(np.asarray(X).shape[0])
    chunk = max(1, min(int(chunk), n))
    n_chunks = max(1, -(-n // chunk))
    Xp = np.zeros(n_chunks * chunk + 2 * c, dtype=np.float64)
    Xp[c:c + n] = X
    return n, chunk, n_chunks, offset.transform_eta(Xp, log_d_src)


def _finish(locs, disps, n: int, log_d_tgt: float):
    loc = np.concatenate(locs)[:n]
    return (offset.offset_loc(loc, log_d_tgt).astype(np.float32),
            np.concatenate(disps)[:n].astype(np.float32))


@torch.no_grad()
def predict_chrom_eta(ladder: EtaLadder, X: np.ndarray, log_d_src: float, log_d_tgt: float,
                      theta: torch.Tensor, device, chunk: int = lmodel.CHUNK):
    """Row 1: (loc, disp) float32[n] for a whole chromosome X, one theta [n_theta]; as
    `ladder.model.predict_chrom` with eta in and loc + log d' out."""
    c = ladder.context
    n, chunk, n_chunks, xp = _padded_eta(X, c, chunk, log_d_src)
    theta = theta.reshape(1, -1).to(device=device, dtype=torch.float32)
    locs, disps = [], []
    for b0 in range(0, n_chunks, lmodel.BATCH_CHUNKS):
        idx = range(b0, min(b0 + lmodel.BATCH_CHUNKS, n_chunks))
        xt = torch.from_numpy(np.stack([xp[i * chunk: i * chunk + chunk + 2 * c] for i in idx]))
        xt = xt.to(device)
        loc, disp = ladder.form(xt, theta.expand(xt.shape[0], -1))
        locs.append(loc.reshape(-1).float().cpu().numpy())
        disps.append(disp.reshape(-1).float().cpu().numpy())
    return _finish(locs, disps, n, log_d_tgt)


@torch.no_grad()
def predict_chrom_bins_eta(ladder: EtaLadder, X: np.ndarray, log_d_src: float, log_d_tgt: float,
                           cov: torch.Tensor, device, average: bool = False,
                           chunk: int = lmodel.CHUNK, level_cache: dict | None = None):
    """Row 2: (loc, disp) float32[n]; as `ladder.model.predict_chrom_bins` (no shuffled-bin twin)
    with eta in and loc + log d' out. `level_cache`: t118's semantics, for ONE fixed (X, log d_src)."""
    c = ladder.context
    n, chunk, n_chunks, xp = _padded_eta(X, c, chunk, log_d_src)
    if level_cache:
        if level_cache["n_padded"] != xp.shape[0]:
            raise ValueError(f"level_cache holds {level_cache['n_padded']} padded bins, "
                             f"this X has {xp.shape[0]}")
        if level_cache.get("log_d_src", float(log_d_src)) != float(log_d_src):
            raise ValueError("level_cache was filled for another source depth")
        uniq, inv = level_cache["uniq"], level_cache["inv"]
    else:
        uniq, inv = np.unique(xp, return_inverse=True)
        if level_cache is not None:
            level_cache.update(n_padded=xp.shape[0], uniq=uniq, log_d_src=float(log_d_src),
                               inv=np.asarray(inv, dtype=np.int32 if uniq.size < 2**31 else np.int64))
    theta_u = lmodel.theta_levels(ladder, torch.from_numpy(uniq).to(device),
                                  cov.to(device=device, dtype=torch.float32), average)
    inv_t = torch.from_numpy(np.asarray(inv, dtype=np.int64).reshape(-1)).to(device)
    offs = torch.arange(chunk + 2 * c, device=device)
    locs, disps = [], []
    for b0 in range(0, n_chunks, lmodel.BATCH_CHUNKS):
        idx = range(b0, min(b0 + lmodel.BATCH_CHUNKS, n_chunks))
        xt = torch.from_numpy(np.stack([xp[i * chunk: i * chunk + chunk + 2 * c] for i in idx]))
        xt = xt.to(device)
        pos = torch.tensor([i * chunk for i in idx], device=device)[:, None] + offs
        loc, disp = ladder.form(xt, theta_u[inv_t[pos]])
        locs.append(loc.reshape(-1).float().cpu().numpy())
        disps.append(disp.reshape(-1).float().cpu().numpy())
    return _finish(locs, disps, n, log_d_tgt)


@torch.no_grad()
def predict_window_eta(ladder: EtaLadder, X_window_haloed: np.ndarray, log_d_src: float,
                       log_d_tgt: float, cond: torch.Tensor, device, average: bool = False):
    """(loc, disp) float32[W] for one haloed window X [W + 2c] (zero-padded outside the chromosome).

    `cond`: row 1, theta [n_theta]; row 2, cov [1, 40] (or [k, 40] with `average=True`, the nocov
    map), evaluated by `theta_levels` at the window's distinct eta values."""
    c = ladder.context
    X = np.asarray(X_window_haloed, dtype=np.float64).reshape(-1)
    if X.shape[0] <= 2 * c:
        raise ValueError(f"window of {X.shape[0]} bins holds no bin beyond the 2 x {c} halo")
    xp = offset.transform_eta(X, log_d_src)
    xt = torch.from_numpy(xp).to(device)[None]
    if ladder.per_bin:
        uniq, inv = np.unique(xp, return_inverse=True)
        theta_u = lmodel.theta_levels(ladder, torch.from_numpy(uniq).to(device),
                                      cond.to(device=device, dtype=torch.float32), average)
        theta = theta_u[torch.from_numpy(np.asarray(inv, dtype=np.int64).reshape(-1)).to(device)]
        loc, disp = ladder.form(xt, theta[None])
    else:
        theta = cond.reshape(1, -1).to(device=device, dtype=torch.float32)
        loc, disp = ladder.form(xt, theta)
    return _finish([loc.reshape(-1).float().cpu().numpy()],
                   [disp.reshape(-1).float().cpu().numpy()], X.shape[0] - 2 * c, log_d_tgt)


class EtaPredictor(lmodel.Predictor):
    """A trained t125 run, ready to predict. Built by `load_run`."""

    def __init__(self, ladder: EtaLadder, encoder: encoding.Encoder, config: dict, corpus,
                 device, step: int, val_nll: float):
        if config["model"] not in MODELS:
            raise ValueError(f"model {config['model']!r} not in {MODELS} (t125 has no ids twin)")
        super().__init__(ladder, encoder, config, corpus, device, step, val_nll)
        self.log_d = {k: float(v) for k, v in config["log_depth"].items()}

    def cov_pids(self, cov_src_pid: str, cov_tgt_pid: str) -> tuple[str, str]:
        return cov_src_pid, cov_tgt_pid

    def _cond(self, cov_src_pid: str, cov_tgt_pid: str):
        if self.per_bin:
            return self._cov_t(cov_src_pid, cov_tgt_pid)
        return self._theta_t(cov_src_pid, cov_tgt_pid), False

    def predict(self, x_src_pid: str, cov_src_pid: str, cov_tgt_pid: str, chrom: str):
        """(loc, disp) float32[n] over the whole chromosome of `x_src_pid`; d = the source's depth,
        d' = `cov_tgt_pid`'s (shuffle: the wrong target's; swap: C' = C so d' = d)."""
        X = self.corpus.get(x_src_pid, SPACE, chrom)
        ld_s, ld_t = self.log_d[x_src_pid], self.log_d[cov_tgt_pid]
        cond, avg = self._cond(cov_src_pid, cov_tgt_pid)
        if self.per_bin:
            return predict_chrom_bins_eta(self.ladder, X, ld_s, ld_t, cond, self.device, average=avg)
        return predict_chrom_eta(self.ladder, X, ld_s, ld_t, cond, self.device)

    def predict_window(self, x_src_pid: str, cov_src_pid: str, cov_tgt_pid: str, chrom: str,
                       start: int, stop: int):
        """(loc, disp) float32[stop - start]; the halo is zero-padded as `Corpus.window` pads."""
        c = self.context
        Xw = self.corpus.window(x_src_pid, SPACE, chrom, int(start) - c, int(stop) + c)
        cond, avg = self._cond(cov_src_pid, cov_tgt_pid)
        return predict_window_eta(self.ladder, Xw, self.log_d[x_src_pid], self.log_d[cov_tgt_pid],
                                  cond, self.device, average=avg)


def load_run(run_dir, data_dir, covariates_tsv, manifest_tsv, device="auto") -> EtaPredictor:
    """The EtaPredictor of one run directory (`<run_dir>/ckpt.pt`, keys as t118's); a covariates /
    manifest md5 that differs from the run's config is a warning, not an error."""
    run_dir = Path(run_dir)
    dev = lmodel.pick_device(device)
    ckpt = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    if cfg.get("space", SPACE) != SPACE:
        raise ValueError(f"{run_dir}: space {cfg['space']!r}, t125 part 1 is counts only")
    for path, key in ((covariates_tsv, "covariates_md5"), (manifest_tsv, "manifest_md5")):
        if path is not None and cfg.get(key) and lmodel.md5_path(path) != cfg[key]:
            warnings.warn(f"{path}: md5 differs from the run's {key}; using the checkpoint's "
                          f"encoder")
    stats = {"knots_x": np.asarray(ckpt["stats"]["knots_x"], dtype=np.float64),
             "n0": float(ckpt["stats"]["n0"]), "sigma0": float(ckpt["stats"]["sigma0"])}
    ladder = EtaLadder(cfg["rung"], stats, n_cov=int(cfg.get("n_cov", encoding.N_COV)))
    ladder.g.load_state_dict(ckpt["g"])
    ladder.form.load_state_dict(ckpt["form"])
    enc = encoding.Encoder.from_state(ckpt["encoder"])
    corpus = data.Corpus(data_dir)
    return EtaPredictor(ladder, enc, cfg, corpus, dev, int(ckpt["step"]), float(ckpt["val_nll"]))

