"""t128 rung D v2 — train one symmetric encoder-decoder run (rungs DS / DSBOX), plan/T126_ENCDEC_D2.md.

`run_training_ds(...)` is `t126.train_ded.run_training_ded` with the symmetric model of
`ladder/fforms/form_ds.py` and `ladder.model.GSym`. For a sampled base<->arm pair (X, c) -> (X', c'),
with cc = C_clean of c (the same for c'), Z = enc(X; g(c, cc)) and Z' = enc(X'; g(c', cc)):

    L1 = NLL(X' | dec(Z; g(cc, c'))) + NLL(X | dec(Z'; g(cc, c)))      cross
    L2 = NLL(X | dec(Z; g(cc, c)))   + NLL(X' | dec(Z'; g(cc, c')))    round trip
    L3 = mse(Z, Z') over the masked centre W bins                       latent agreement
    total = L1 [loss L12, L123] + L2 + lam * L3 [loss L123] + beta(step) * KL [vae]

Deterministic settings use Z = mu_z; VAE settings Z = mu_z + sigma_z * eps (`torch.randn_like`, so
the run's `torch.manual_seed` fixes eps), KL to N(0, 1) on the masked centre bins of Z and Z',
weighted by beta(step) = min(1, step / 1000). C_clean: form A (rung DS) starts at zeros, form D
(rung DSBOX) at the base's encoded knobs inside the box [min, max] over the fit pids (PI 2026-10-21).

Validation (PI ruling (iii), 2026-10-21): every eval computes, on the whole of chr22 with the
deterministic forward (Z = mu_z), the setting's OWN NLL terms (`val_nll_rule`: "L1+L2 NLL" or
"L2 NLL"; no KL, no lam * mse), mean over the trained pairs -> `val_nll`, which selects the best
step; and t118's cross NLL X -> X' (`ladder.train.validate`, unchanged) -> `val_cross_nll`,
logged only. The no-covariates twin uses the mean theta over the trained pairs in both.

Everything else is t118 row 1's (lr 1e-3, 100-step warmup, grad clip 1.0, batch 32, 4000 steps,
eval every 250, patience 4, best state saved), the run dir is t126's, and `ckpt.pt` is readable by
`ladder.model.load_run` (rung "DS" / "DSBOX"), so `ladder/score.py trained` / `law` score it.

Phase 2 (plan/T126_ENCDEC_D2.md "Phase 2: PI decisions, 2026-10-25"; phase-1 paths unchanged):
C_clean form "F" = the base's encoded knobs, frozen (out of the optimizer); loss "L12A" (rung
DSANC) = L1 + L2 + L_anchor, L_anchor = NB NLL of the base counts given the arm's encoder head,
the middle Z = log1p of that head's mean; val rule "L1+L2+anchor NLL"; `setting["max_steps"]`
(12,000) replaces DEFAULTS' 4000; every run also writes `train_terms.tsv`.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ladder import base, data, encoding, model, pairs  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from ladder.fforms.form_ds import CONTEXT, N_ASSAY, N_KNOB  # noqa: E402
from t126.train_ded import PairSampler  # noqa: E402
from t128.pairs import HOLDOUT  # noqa: E402

SPACE = "counts"
KL_WARMUP_STEPS = 1000
#: one f's context; the centre W bins of Z [B, W + 60] are Z[:, CTX1:-CTX1]
CTX1 = 30
LOSSES = ("L12", "L2", "L123", "L12A")
#: "A" free, zero init; "D" boxed, base init (rung DSBOX); "F" frozen at the base (phase 2)
CCLEAN_FORMS = ("A", "D", "F")
VAL_RULES = ("L1+L2 NLL", "L2 NLL", "L1+L2+anchor NLL")
LOG_HEADER = "step\tloss\tmse_z\tkl\tval_nll\tval_cross_nll\tseconds"
TERMS_HEADER = "step\tL1\tL2\tanchor\tmse_z\tkl"


def cclean_tables(enc: encoding.Encoder, rows, fit_pids):
    """(init, lo, hi), float32 [7, 13] each: per assay with a fit pid, init = the encoded knobs of
    the base product of that assay's track, lo / hi = min / max of the fit pids' encoded knobs;
    0 for absent assays."""
    init = torch.zeros(N_ASSAY, N_KNOB)
    lo = torch.zeros(N_ASSAY, N_KNOB)
    hi = torch.zeros(N_ASSAY, N_KNOB)
    by_pid = {r["pid"]: r for r in rows}
    for a, assay in enumerate(encoding.ASSAYS):
        pids = [p for p in fit_pids if enc.rows[p]["assay"] == assay]
        if not pids:
            continue
        k = torch.from_numpy(np.stack([enc.encode(p)[:N_KNOB] for p in pids]))
        lo[a], hi[a] = k.min(0).values, k.max(0).values
        bases = sorted({by_pid[p]["pid"] for p in pids if by_pid[p]["arm"] == "base"})
        if len(bases) != 1:
            raise ValueError(f"assay {assay}: {len(bases)} base pids among the fit pids, need 1")
        init[a] = torch.from_numpy(enc.encode(bases[0])[:N_KNOB])
    return init, lo, hi


def beta(step: int) -> float:
    return min(1.0, float(step) / KL_WARMUP_STEPS)


def kl_gauss(mu, log_sigma, mask=None) -> torch.Tensor:
    """Masked mean over bins of KL(N(mu, sigma^2) || N(0, 1))."""
    per = 0.5 * (mu ** 2 + torch.exp(2.0 * log_sigma) - 1.0 - 2.0 * log_sigma)
    if mask is None:
        return per.mean()
    m = mask.to(per.dtype)
    return (per * m).sum() / m.sum().clamp(min=1.0)


def sym_loss(ladder: model.Ladder, cov, xs, xt, ys, yt, mask, setting: dict, step: int,
             src_is_base=None):
    """(total, terms) for a batch: cov [B, 40] = (c, c'), xs / xt [B, W+120], ys / yt / mask [B, W].

    Loss "L12A" (phase 2, rung DSANC) needs `src_is_base` Bool[B] (the pair's direction is
    base_to_arm): Z = middle(encode_nb loc), and L_anchor = the masked mean NB NLL of the BASE
    counts given the ARM's encoder head; total = L1 + L2 + L_anchor. Other losses ignore it."""
    form, g = ladder.form, ladder.g
    loss = setting["loss"]
    if loss not in LOSSES:
        raise ValueError(f"loss {loss!r} not in {LOSSES}")
    anchored = loss == "L12A"
    if anchored and src_is_base is None:
        raise ValueError("loss L12A needs src_is_base")
    if anchored and not hasattr(form, "encode_nb"):
        raise ValueError(f"loss L12A needs a form with encode_nb (rung DSANC), not {ladder.rung}")
    vae = bool(setting["vae"])
    n = g.n_cov
    c, ct = cov[:, :n], cov[:, n:]
    cc = g.cclean_of(c)
    te, tet = g.film(c, cc), g.film(ct, cc)
    td, tdt = g.film(cc, c), g.film(cc, ct)
    if anchored:
        loc_s, disp_s = form.encode_nb(xs, te)
        loc_t, disp_t = form.encode_nb(xt, tet)
        mu, ls = form.middle(loc_s), disp_s
        mut, lst = form.middle(loc_t), disp_t
    else:
        mu, ls = form.encode(xs, te)
        mut, lst = form.encode(xt, tet)
    if vae:
        z = mu + torch.exp(ls) * torch.randn_like(mu)
        zt = mut + torch.exp(lst) * torch.randn_like(mut)
    else:
        z, zt = mu, mut
    terms = {
        "cross_st": ladder.nll(*form.decode(z, tdt), yt, mask),
        "cross_ts": ladder.nll(*form.decode(zt, td), ys, mask),
        "self_s": ladder.nll(*form.decode(z, td), ys, mask),
        "self_t": ladder.nll(*form.decode(zt, tdt), yt, mask),
    }
    terms["L1"] = terms["cross_st"] + terms["cross_ts"]
    terms["L2"] = terms["self_s"] + terms["self_t"]
    ctr = slice(CTX1, -CTX1)
    m = mask.to(z.dtype)
    terms["mse_z"] = (((z[:, ctr] - zt[:, ctr]) ** 2) * m).sum() / m.sum().clamp(min=1.0)
    if vae:
        terms["kl"] = (kl_gauss(mu[:, ctr], ls[:, ctr], mask)
                       + kl_gauss(mut[:, ctr], lst[:, ctr], mask))
    else:
        terms["kl"] = torch.zeros((), dtype=z.dtype, device=z.device)
    terms["beta"] = beta(step)
    if anchored:
        sib = torch.as_tensor(src_is_base, dtype=torch.bool, device=z.device).reshape(-1, 1)
        anchor_bins = torch.where(sib, ladder.nll_bins(loc_t[:, ctr], disp_t[:, ctr], ys),
                                  ladder.nll_bins(loc_s[:, ctr], disp_s[:, ctr], yt))
        mb = mask.to(anchor_bins.dtype)
        terms["anchor"] = (anchor_bins * mb).sum() / mb.sum().clamp(min=1.0)
    else:
        terms["anchor"] = torch.zeros((), dtype=z.dtype, device=z.device)
    total = terms["L2"]
    if loss in ("L12", "L123", "L12A"):
        total = total + terms["L1"]
    if loss == "L123":
        total = total + float(setting["lam"]) * terms["mse_z"]
    if anchored:
        total = total + terms["anchor"]
    if vae:
        total = total + terms["beta"] * terms["kl"]
    return total, terms


@torch.no_grad()
def _chrom_mu(ladder, X: np.ndarray, theta_e: torch.Tensor, device) -> torch.Tensor:
    """mu_z over a whole chromosome in chunks: [n_chunks, chunk + 60] (Z with its decoder halo)."""
    c = ladder.context
    n = int(np.asarray(X).shape[0])
    chunk = max(1, min(model.CHUNK, n))
    n_chunks = max(1, -(-n // chunk))
    Xp = np.zeros(n_chunks * chunk + 2 * c, dtype=np.float64)
    Xp[c:c + n] = X
    xp = model.transform_x(Xp, ladder.space)
    theta_e = theta_e.reshape(1, -1).to(device=device, dtype=torch.float32)
    out = []
    for b0 in range(0, n_chunks, model.BATCH_CHUNKS):
        idx = range(b0, min(b0 + model.BATCH_CHUNKS, n_chunks))
        xb = torch.from_numpy(np.stack([xp[i * chunk: i * chunk + chunk + 2 * c]
                                        for i in idx])).to(device)
        out.append(ladder.form.encode(xb, theta_e.expand(xb.shape[0], -1))[0])
    return torch.cat(out)


@torch.no_grad()
def _chrom_nb(ladder, X: np.ndarray, theta_e: torch.Tensor, device):
    """The encoder NB head (`form.encode_nb`) over a whole chromosome, chunked as `_chrom_mu`:
    (loc, disp) [n_chunks, chunk + 60]; the centre bins [:, 30:-30] align with X."""
    c = ladder.context
    n = int(np.asarray(X).shape[0])
    chunk = max(1, min(model.CHUNK, n))
    n_chunks = max(1, -(-n // chunk))
    Xp = np.zeros(n_chunks * chunk + 2 * c, dtype=np.float64)
    Xp[c:c + n] = X
    xp = model.transform_x(Xp, ladder.space)
    theta_e = theta_e.reshape(1, -1).to(device=device, dtype=torch.float32)
    locs, disps = [], []
    for b0 in range(0, n_chunks, model.BATCH_CHUNKS):
        idx = range(b0, min(b0 + model.BATCH_CHUNKS, n_chunks))
        xb = torch.from_numpy(np.stack([xp[i * chunk: i * chunk + chunk + 2 * c]
                                        for i in idx])).to(device)
        loc, disp = ladder.form.encode_nb(xb, theta_e.expand(xb.shape[0], -1))
        locs.append(loc)
        disps.append(disp)
    return torch.cat(locs), torch.cat(disps)


@torch.no_grad()
def _chrom_anchor_nll(ladder, X_arm, theta_e, Y_base, device) -> float:
    """NB NLL of the base's counts `Y_base` given the arm's encoder head, whole chromosome."""
    loc, disp = _chrom_nb(ladder, X_arm, theta_e, device)
    n = int(np.asarray(Y_base).shape[0])
    loc = loc[:, CTX1:-CTX1].reshape(-1)[:n]
    disp = disp[:, CTX1:-CTX1].reshape(-1)[:n]
    y = torch.from_numpy(np.array(Y_base, dtype=np.float32)).to(device)
    return float(ladder.nll(loc, disp, y))


@torch.no_grad()
def _chrom_nll(ladder, z: torch.Tensor, theta_d: torch.Tensor, Y: np.ndarray, device) -> float:
    n = int(np.asarray(Y).shape[0])
    theta_d = theta_d.reshape(1, -1).to(device=device, dtype=torch.float32)
    locs, disps = [], []
    for b0 in range(0, z.shape[0], model.BATCH_CHUNKS):
        zb = z[b0:b0 + model.BATCH_CHUNKS]
        loc, disp = ladder.form.decode(zb, theta_d.expand(zb.shape[0], -1))
        locs.append(loc.reshape(-1))
        disps.append(disp.reshape(-1))
    y = torch.from_numpy(np.array(Y, dtype=np.float32)).to(device)
    return float(ladder.nll(torch.cat(locs)[:n], torch.cat(disps)[:n], y))


def validate_own(ladder, corpus, train_pairs, cov_val: torch.Tensor, device, rule: str,
                 average_theta: bool = False) -> float:
    """Mean over the trained pairs of the setting's own NLL terms on the whole of chr22,
    deterministic (Z = mu_z): "L1+L2 NLL" = cross_st + cross_ts + self_s + self_t, "L2 NLL" =
    self_s + self_t. "L1+L2+anchor NLL" (phase 2, rung DSANC) = the "L1+L2 NLL" value + the NB NLL
    of the base's chr22 counts given the arm's encoder head (`_chrom_nb`; the arm is the pair's
    non-base pid). `average_theta` (the nocov twin): one theta for every encode and decode, the
    mean over `cov_val` (`model.nocov_theta`), as the twin's Predictor."""
    if rule not in VAL_RULES:
        raise ValueError(f"val_nll_rule {rule!r}")
    chrom = pairs.VAL_CHROMS[0]
    form, g, n = ladder.form, ladder.g, ladder.g.n_cov
    was_training = ladder.training
    ladder.eval()
    out = []
    with torch.no_grad():
        avg = (form.split_theta(model.nocov_theta(ladder, cov_val).reshape(1, -1))
               if average_theta else None)
        for i, p in enumerate(train_pairs):
            X = corpus.get(p["source_pid"], SPACE, chrom)
            Y = corpus.get(p["target_pid"], SPACE, chrom)
            if avg is not None:
                te = tet = avg[0][0]
                td = tdt = avg[1][0]
            else:
                cv = cov_val[i:i + 1]
                c, ct = cv[:, :n], cv[:, n:]
                cc = g.cclean_of(c)
                te, tet = g.film(c, cc)[0], g.film(ct, cc)[0]
                td, tdt = g.film(cc, c)[0], g.film(cc, ct)[0]
            z, zt = _chrom_mu(ladder, X, te, device), _chrom_mu(ladder, Y, tet, device)
            v = _chrom_nll(ladder, z, td, X, device) + _chrom_nll(ladder, zt, tdt, Y, device)
            if rule in ("L1+L2 NLL", "L1+L2+anchor NLL"):
                v += _chrom_nll(ladder, z, tdt, Y, device) + _chrom_nll(ladder, zt, td, X, device)
            if rule == "L1+L2+anchor NLL":
                if p["direction"] == "base_to_arm":          # arm = target, base = source
                    v += _chrom_anchor_nll(ladder, Y, tet, X, device)
                else:                                        # arm = source, base = target
                    v += _chrom_anchor_nll(ladder, X, te, Y, device)
            out.append(v)
    ladder.train(was_training)
    return float(np.mean(out))


def _table(t: torch.Tensor) -> dict:
    t = t.detach().double().cpu()
    return {a: t[i].tolist() for i, a in enumerate(encoding.ASSAYS)}


def run_training_ds(manifest, covariates, data_dir, runs_dir, setting: dict, g, model_name, seed,
                    holdout=HOLDOUT, max_steps=None, eval_every=None, device="auto",
                    cov_rows=None, **overrides):
    """Train one DS / DSBOX / DSANC run into `<runs_dir>/<arm>_<g>_counts_<model>_s<seed>/`; skips
    if TRAIN_DONE.

    Phase 2: `setting["max_steps"]`, when present, replaces DEFAULTS' 4000 (the `max_steps`
    argument still wins); C_clean form "F" starts at the base's encoded knobs and is frozen (out of
    the optimizer); `train_terms.tsv` logs the mean L1, L2, anchor, mse_z and kl per eval."""
    t0 = time.time()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if model_name not in ("real", "nocov"):
        raise ValueError(f"model {model_name!r} not in (real, nocov)")
    if setting["loss"] not in LOSSES:
        raise ValueError(f"loss {setting['loss']!r} not in {LOSSES}")
    if setting["cclean"] not in CCLEAN_FORMS:
        raise ValueError(f"cclean {setting['cclean']!r} not in {CCLEAN_FORMS}")
    arm, rung, rule = setting["arm"], setting["rung"], setting["val_nll_rule"]
    if rule not in VAL_RULES:
        raise ValueError(f"val_nll_rule {rule!r} not in {VAL_RULES}")
    seed = int(seed)
    cfg = dict(ltrain.DEFAULTS)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"unknown training settings: {sorted(unknown)}")
    cfg.update(overrides)
    cfg["max_steps"] = setting.get("max_steps", cfg["max_steps"])
    if max_steps is not None:
        cfg["max_steps"] = int(max_steps)
    if eval_every is not None:
        cfg["eval_every"] = int(eval_every)
    name = f"{arm}_{g}_{SPACE}_{model_name}_s{seed}"
    run_dir = Path(runs_dir) / name
    if (run_dir / "TRAIN_DONE").is_file():
        print(f"{name}: TRAIN_DONE present, nothing to do")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = model.pick_device(device)

    rows = pairs.read_manifest(manifest)
    if cov_rows is None:
        cov_rows = encoding.read_covariates(covariates)
    holdout = [[str(a), str(lv)] for a, lv in holdout]
    tp = pairs.train_pairs(rows, g, holdout)
    fit = pairs.fit_pids(rows, g, holdout)
    if len(tp) < 2:
        raise ValueError(f"g {g!r}: {len(tp)} training pairs")
    corpus = data.Corpus(data_dir)
    if not corpus.is_cache:
        warnings.warn(f"{data_dir} is a products dir of npz files, not the memory-mapped cache: "
                      f"meant for synthetic data only", stacklevel=2)
    enc = encoding.Encoder.fit(cov_rows, fit)
    perm = pairs.ids_permutation(len(tp), seed)
    cov_all = np.stack([enc.encode_pair(p["source_pid"], p["target_pid"]) for p in tp])
    cov_all_t = torch.from_numpy(cov_all).to(dev)
    cov_val = cov_all_t

    chroms, lens = ltrain.training_chroms(corpus, tp[0]["source_pid"], SPACE)
    xs = ltrain.knot_sample(corpus, fit, SPACE, chroms, lens, cfg["knot_sample"], cfg["knot_block"])
    stats = {"knots_x": base.knot_stats(xs), "n0": float(cfg["n0"]), "sigma0": float(cfg["sigma0"])}
    del xs

    torch.manual_seed(seed)
    np.random.seed(seed)
    ladder = model.Ladder(rung, SPACE, stats).to(dev)
    assert ladder.context == CONTEXT == 60
    base_knobs, lo, hi = cclean_tables(enc, rows, fit)
    box = bool(ladder.g.box)
    frozen = setting["cclean"] == "F"
    if frozen and box:
        raise ValueError(f"cclean F (frozen) needs an unboxed rung, not {rung}")
    if box:
        ladder.g.set_box(lo, hi)
        init = base_knobs                               # form D: the base's covariates
    elif frozen:
        init = base_knobs                               # form F: the base's covariates, frozen
    else:
        init = torch.zeros(N_ASSAY, N_KNOB)             # form A: zeros (PI ruling (ii))
    ladder.g.set_cclean_init(init)
    if frozen:
        ladder.g.cclean.requires_grad_(False)           # Decision (v): out of the optimizer
    sampler = PairSampler(corpus, SPACE, tp, chroms, lens, cfg["window"], ladder.context, seed,
                          model_name, perm)
    opt = torch.optim.Adam([p for p in ladder.parameters() if p.requires_grad], lr=cfg["lr"])
    warm = max(1, int(cfg["warmup"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))

    loss_txt = {"L12": "L1 + L2", "L2": "L2", "L123": "L1 + L2 + lam*L3",
                "L12A": "L1 + L2 + L_anchor"}[setting["loss"]]
    if setting["vae"]:
        loss_txt += " + beta(step)*KL"
    config = {
        "run_name": name, "rung": rung, "arm": arm, "setting": dict(setting),
        "cclean_form": setting["cclean"], "cclean_frozen": frozen, "vae": bool(setting["vae"]),
        "lam": float(setting["lam"]), "g": g, "space": SPACE,
        "model": model_name, "seed": seed, **cfg, "device": str(dev), "n_cov": encoding.N_COV,
        "cov_fields": list(encoding.FIELDS), "context": ladder.context, "n_theta": ladder.n_theta,
        "train_chroms": chroms, "val_chroms": list(pairs.VAL_CHROMS),
        "score_chroms": list(pairs.SCORE_CHROMS),
        "sampling": "pair uniform, chromosome proportional to length, start uniform",
        "nocov_rng": "default_rng([seed, 7121]) per sample per step",
        "knots": "base.knot_stats on knot_sample bins in knot_block blocks, default_rng(0)",
        "loss": f"{setting['loss']}: {loss_txt}; L1 = NLL(X'|dec(Z;g(cc,c'))) + "
                "NLL(X|dec(Z';g(cc,c))), L2 = NLL(X|dec(Z;g(cc,c))) + NLL(X'|dec(Z';g(cc,c'))), "
                "L3 = mse(Z,Z') on the masked centre bins, KL to N(0,1) on the masked centre bins",
        "kl_warmup_steps": KL_WARMUP_STEPS,
        "val_nll_rule": rule,
        "val_cross_nll": "t118 cross NLL X->X' on chr22 (ladder.train.validate), logged only",
        "cclean_init": _table(init),
        "cclean_init_decoded": ladder.g.cclean_table(),
        "git_sha": ltrain.git_sha(), "manifest": str(manifest),
        "manifest_md5": model.md5_path(manifest), "covariates": str(covariates),
        "covariates_md5": model.md5_path(covariates) if covariates else None,
        "data_dir": str(data_dir), "fit_pids": fit, "train_pairs": tp,
        "ids_perm": [int(v) for v in perm], "knots_x": [float(v) for v in stats["knots_x"]],
    }
    if setting["loss"] == "L12A":
        config["anchor"] = "NLL(X_base | encode_nb(x_arm; g(c_arm, C_clean)))[centre bins]"
    if rung == "DSANC":
        config["middle"] = "log1p(clamp(exp(loc_e), 1e-4, 1e6))"
    if box:
        config["box_lo"], config["box_hi"] = _table(lo), _table(hi)
    if holdout:
        config["holdout"] = holdout
        config["heldout_pairs"] = pairs.heldout_pairs(rows, g, holdout)
    (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")

    t_setup = time.time() - t0
    t1 = time.time()
    log = []
    nocov = model_name == "nocov"

    def evaluate():
        own = validate_own(ladder, corpus, tp, cov_val, dev, rule, nocov)
        cross = ltrain.validate(ladder, corpus, tp, cov_val, SPACE, dev, nocov)
        return own, cross

    val, vcross = evaluate()
    nan = float("nan")
    log.append((0, nan, nan, nan, val, vcross, time.time() - t1))
    tlog = [(0, nan, nan, nan, nan, nan)]
    best = {"val": val, "cross": vcross, "step": 0, "g": copy.deepcopy(ladder.g.state_dict()),
            "form": copy.deepcopy(ladder.form.state_dict())}
    bad, losses, mses, kls, step = 0, [], [], [], 0
    l1s, l2s, anchors = [], [], []
    ladder.train()
    for step in range(1, cfg["max_steps"] + 1):
        pi, ci, xb, xtb, ysb, ytb, mb = sampler.draw(cfg["batch"])
        cov = cov_all_t[torch.from_numpy(ci).to(dev)]
        src_is_base = torch.tensor([tp[i]["direction"] == "base_to_arm" for i in pi], device=dev)
        t = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
        loss, terms = sym_loss(ladder, cov, t(xb), t(xtb), t(ysb), t(ytb), t(mb), setting, step,
                               src_is_base=src_is_base)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name}: non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ladder.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()
        losses.append(float(loss))
        mses.append(float(terms["mse_z"]))
        kls.append(float(terms["kl"]))
        l1s.append(float(terms["L1"]))
        l2s.append(float(terms["L2"]))
        anchors.append(float(terms["anchor"]))
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            val, vcross = evaluate()
            log.append((step, float(np.mean(losses)), float(np.mean(mses)), float(np.mean(kls)),
                        val, vcross, time.time() - t1))
            tlog.append((step, float(np.mean(l1s)), float(np.mean(l2s)), float(np.mean(anchors)),
                         float(np.mean(mses)), float(np.mean(kls))))
            losses, mses, kls = [], [], []
            l1s, l2s, anchors = [], [], []
            if val < best["val"]:
                best = {"val": val, "cross": vcross, "step": step,
                        "g": copy.deepcopy(ladder.g.state_dict()),
                        "form": copy.deepcopy(ladder.form.state_dict())}
                bad = 0
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
    t_train = time.time() - t1

    ckpt = {"g": {k: v.cpu() for k, v in best["g"].items()},
            "form": {k: v.cpu() for k, v in best["form"].items()},
            "stats": {"knots_x": [float(v) for v in stats["knots_x"]], "n0": stats["n0"],
                      "sigma0": stats["sigma0"]},
            "encoder": enc.state(), "config": config, "step": int(best["step"]),
            "val_nll": float(best["val"]), "val_cross_nll": float(best["cross"])}
    tmp = run_dir / "ckpt.pt.tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, run_dir / "ckpt.pt")
    with open(run_dir / "train_log.tsv", "w", encoding="utf-8") as fh:
        fh.write(LOG_HEADER + "\n")
        for s, lo_, ms, kl, v, vc, sec in log:
            fh.write(f"{s}\t{lo_!r}\t{ms!r}\t{kl!r}\t{v!r}\t{vc!r}\t{sec:.3f}\n")
    with open(run_dir / "train_terms.tsv", "w", encoding="utf-8") as fh:
        fh.write(TERMS_HEADER + "\n")
        for row in tlog:
            fh.write("\t".join([str(row[0])] + [repr(v) for v in row[1:]]) + "\n")
    ladder.g.load_state_dict(best["g"])
    cclean = {"assays": ladder.g.cclean_table(), "init": config["cclean_init"], "box": box,
              "frozen": frozen}
    (run_dir / "cclean.json").write_text(json.dumps(cclean, indent=1) + "\n")
    timing = {"setup": t_setup, "train": t_train, "total": time.time() - t0,
              "peak_rss_mb": ltrain.peak_rss_mb(), "steps_run": step,
              "best_step": int(best["step"]), "val_nll": float(best["val"]),
              "val_cross_nll": float(best["cross"])}
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    (run_dir / "TRAIN_DONE").write_text(f"{name} best_step {best['step']} val_nll {best['val']!r}\n")
    print(f"{name}: best_step {best['step']} val_nll {best['val']:.6g} "
          f"steps {step} train {t_train:.1f}s total {timing['total']:.1f}s")
    return run_dir
