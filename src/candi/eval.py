"""candi measurement stack — M1/M2/M3 + the S14 depth counterfactual on real batches.

VENDORED (COPY+EDIT) of sandbox/diagnostics/dual_conditioning_real/metrics_real.py:1-1001,
plus _build_vb_natural_missing_meta VERBATIM out of sandbox/train.py:118-142.

Readouts run on real `CandiKitH5Dataset` batches and use ONLY the primitives in `candi.metrics`
(`nb_crps`, `nb_quantile`, `ece`/`calibration_pit_curve`, `spearman`, `pearson`, `r2`, `_cos_dist`)
plus the p-from-true-mu-in-float64 convention.

- **M1 (health)** — imp/den count Spearman+Pearson, NB CRPS, PIT-ECE, per-assay honest marginal +
  oracle-scale decomposition, per-position encoder eff-rank.
- **M2 (depth / run_type / read_length)** — the counterfactual-prompt FLIP test on the held-out
  imputation targets (target absent from the T_ input => the prompt is the only channel), the told-depth
  sweep with clamp telemetry, and the sentinel-free cross-target metadata ablation.
- **M3 (invariance)** — same region encoded at input DSF in {1,2,4,8} (each with its TRUE x_meta) ->
  within/between latent cos-dist ratio, guarded by encoder eff-rank>1.
- **S14** — each told depth scored against its OWN ground truth `counts_dsf{k}`.

THE ASSAY ORDER IS NEVER DECLARED HERE. `build_eval_units` returns it alongside the units, read from
the h5 attrs via the dataset, and the caller threads it into every labelled instrument.

The encoder latent `z` is invariant to `y_meta`, so it is cached per eval unit and prompt variants only
re-run the (cheap) decoder.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

from candi.batch import CLOZE, MISSING, make_masker, prepare_masked_batch
from candi.dataset import CandiKitH5Dataset, base_cell_type, cell_id_map
from candi.metrics import (
    nb_crps, nb_quantile, ece, calibration_pit_curve, spearman, pearson, r2, _cos_dist, P_EPS,
)
from candi.model import forward_full
from candi.precision import no_autocast

RUN_TYPE_ROW, DEPTH_ROW, READLEN_ROW = 3, 0, 2
OBSERVED_READLENS = (30.0, 36.0, 76.0, 100.0, 101.0)

# Keys emitted ONLY under include_deprecated=True. Each one is a measurement that was made, audited and
# found not to support the claim it was reporting; they ship with their verdict so a reader who finds one
# in a results JSON cannot mistake it for evidence.
DEPRECATED_VERDICTS = {
    "read_length": "7/12 flips land outside per-assay training support -> measures OOD extrapolation, "
                   "not read-length steering.",
    "null": "shuffled-depth null is a mathematical NO-OP: y_meta_imp is one [4,F] tensor broadcast over "
            "the batch, so base_d[perm] == base_d bitwise. A real null must permute ACROSS targets.",
    "null_clustered": "same no-op as `null`, target-clustered.",
    "frac_min_at_true": "scores every told depth against the fixed dsf1 target, so any mu-decreasing "
                        "model satisfies it (0.7588 vs 0.7597 between arms). Superseded by S14.",
    "direction": "position-level bootstrap CI — ~24x too narrow; positions within a target are not "
                 "independent draws. Use the *_clustered keys.",
    "overall": "position-level bootstrap CI — ~24x too narrow. Use overall_clustered.",
    "single": "position-level bootstrap CI — ~24x too narrow. Use single_clustered.",
    "paired": "position-level bootstrap CI — ~24x too narrow. Use paired_clustered.",
    "median_eta_slope": "offset-free residual; decided by the sign of ~1e-17 float noise under "
                        "offset-ON. total_slope = beta + eta_slope, so eta_slope ~ 0 under a correct "
                        "offset is arithmetically right, not a failure.",
    "offset_independent": "boolean read of median_eta_slope > 0 — i.e. of float noise.",
    "frac_direction": "strict `>` on mean_delta, so exact ties report 0.0% correct rather than "
                      "'no signal'.",
}


# ---------------------------------------------------------------------------
# lifted VERBATIM out of sandbox/train.py:118-142 (the only reason metrics_real.py imported sandbox.train)
# ---------------------------------------------------------------------------

def _build_vb_natural_missing_meta(
    t_meta: torch.Tensor,
    vb_meta: torch.Tensor,
    y_avail: torch.Tensor,
    canonical_meta: Optional[torch.Tensor],
) -> torch.Tensor:
    """V/B natural metadata for assays missing in T (y_avail==0); canonical fallback.

    Used when ``use_canonical_missing_meta=False`` (E32 / CANDI v2 default): inject the
    paired V_*/B_* biosample's real covariates at imp-eval slots so depth_offset heads
    see the correct sequencing depth. Falls back to EIC canonical medians when V/B meta
    row 0 is invalid (-1).
    """
    device = t_meta.device
    mixed = t_meta.clone()
    missing = (y_avail.to(device) == 0).unsqueeze(1).expand_as(mixed)
    vb = vb_meta.to(device)
    valid_vb = (vb[:, 0:1, :] != -1.0).expand_as(mixed)
    use_vb = missing & valid_vb
    mixed[use_vb] = vb[use_vb]
    if canonical_meta is not None:
        can_exp = canonical_meta.to(device).unsqueeze(0).expand_as(mixed)
        still_missing = missing & (mixed[:, 0:1, :] == -1.0).expand_as(mixed)
        mixed[still_missing] = can_exp[still_missing]
    return mixed


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def effective_rank(Z: np.ndarray) -> float:
    """Roy-Vetterli effective rank = exp(entropy of the normalized singular-value spectrum)."""
    if Z.ndim != 2 or Z.shape[0] < 2:
        return float("nan")
    Zc = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return float("nan")
    pk = s / s.sum()
    return float(np.exp(-np.sum(pk * np.log(pk))))


def _foreground_mask(target: np.ndarray, fg_frac: float, min_n: int = 5,
                     seed: int = 0) -> Tuple[np.ndarray, bool]:
    """RANK-based foreground: the top `fg_frac` positions by count, not `target >= quantile`.

    The quantile form degenerates on sparse tracks: with 36-67% exact zeros the 98th-percentile of a
    held-out target is often 0, so `target >= 0` selects EVERY position. Measured on the q19 eval:
    2% requested, 23.9% realised, 21.6% of records at 100% -- and those records supplied 90% of the
    bootstrap mass, so the steering tests ran mostly on background.

    Rank selection makes the realised fraction exact by construction. `target >= 1` is then applied as
    a purity filter whenever it still leaves `min_n` positions, so background can never enter while real
    signal is available; an all-zero target keeps the top-k so the mask is never empty.

    Returns `(mask, purity_fallback_fired)`. The flag is True when the `target >= 1` purity filter was
    DROPPED -- it fires on ~18% of real held-out records and the record still reports a healthy-looking
    `n_fg`, so it is propagated into every per-target record and excluded from `_cluster_bootstrap_ci`
    by default.

    Ties at the selection threshold are broken by a SEEDED RANDOM draw. `argsort(kind="stable")` takes
    the LAST index among ties and 87.8% of records break a tie there, giving a mean normalized genomic
    position of 0.618 against an unbiased 0.50.
    """
    n = target.size
    if n == 0:
        return np.zeros(0, bool), False
    k = int(min(n, max(min_n, round(fg_frac * n))))
    m = np.zeros(n, bool)
    jitter = np.random.default_rng(seed).random(n)
    m[np.lexsort((jitter, target))[-k:]] = True
    pos = target >= 1.0
    if int(pos.sum()) >= min(min_n, n):
        return m & pos, False
    return m, True


def _fg_pos_mean(fg: np.ndarray) -> float:
    """Realized mean normalized genomic position of the selected foreground (unbiased = 0.50)."""
    if fg.size < 2 or not bool(fg.any()):
        return float("nan")
    return float(np.mean(np.flatnonzero(fg) / (fg.size - 1.0)))


def _p_from_true_mu(n: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return np.clip(n / (n + mu), P_EPS, 1.0 - P_EPS)


# t22: the bodies moved to `candi.stats` (they are statistics, not measurement, and three
# call sites outside this module want them without wanting an eval harness). The private
# names are kept as aliases so nothing below this line changed at all.
from candi.stats import bootstrap_ci as _bootstrap_ci  # noqa: E402
from candi.stats import cluster_bootstrap_ci as _cluster_bootstrap_ci  # noqa: E402
from candi.stats import sign_test_p as _sign_test_p  # noqa: E402


# ---------------------------------------------------------------------------
# eval units — real eval-chromosome batches with the imputation prompt; cache z + GT
# ---------------------------------------------------------------------------

class EvalUnit:
    """One (T_, V_/B_) eval batch: cached latent + prompt + den/imp masks + GT (all on device)."""

    def __init__(self, z, y_meta_true, den_map, y_data, imp_map=None, y_data_imp=None,
                 y_meta_imp=None, y_avail=None, biosample="", imp_biosample="", log_ref=None):
        self.z = z
        self.y_meta_true = y_meta_true           # [B,4,A] mixed prompt (VB-natural at imp slots)
        self.den_map = den_map                   # [B,L,A] unmasked available T_ (denoise GT = y_data)
        self.y_data = y_data                     # [B,L,A]
        self.imp_map = imp_map                    # [B,L,A] or None
        self.y_data_imp = y_data_imp             # [B,L,A] or None (imp GT)
        self.y_meta_imp = y_meta_imp             # [B,4,A] V/B natural meta or None
        self.y_avail = y_avail                   # [B,A] or None
        self.biosample = biosample
        self.imp_biosample = imp_biosample
        # h74 reference offset [B,L,A] or None. Cached on the unit next to `z` for the same reason:
        # it depends only on the windows and the T_ biosample, so every prompt variant M2 decodes
        # reuses it and the counterfactual isolates the prompt, not the offset.
        self.log_ref = log_ref

    def imp_target_assays(self) -> List[int]:
        if self.imp_map is None:
            return []
        return [a for a in range(self.imp_map.shape[-1]) if bool(self.imp_map[:, :, a].any())]


@torch.no_grad()
def eval_cell_cond(model) -> str:
    """The `cell_cond` an eval dataset must be opened with, read off the MODEL.

    It is read off the checkpoint and never passed in, because the eval prompt has to carry exactly
    the rows the trained embedder expects; a caller free to disagree with the checkpoint is a bug
    waiting to be written.

    The mode is `"id"` for EVERY conditioned arm, the `cell_cond="random"` null included. All arms
    are then scored under one identical protocol, which is what makes the comparison a comparison;
    a null-arm model that learned to ignore row 4 in training is unmoved by being handed a real id
    here, while one that did NOT is exactly what the numbers should expose.

    It lives beside the scorer rather than inside it because the scorer no longer opens its own
    data — see `build_eval_units`.
    """
    return "id" if _model_num_cells(model) > 0 else "off"


def open_h5_eval_dataset(model, h5_path, *, regime: str = "type1", batch_size: int = 4,
                         imp_prefixes=("V_", "B_"), seed: int = 0, reference=None):
    """The h5 eval dataset, opened exactly as `build_eval_units` used to open it.

    Kept whole and kept HERE, in the half of this module that is h5-only, because the store path
    does not go through it: a store declares its pairing in the regime's `eval_pairs` (D31) rather
    than deriving it from a name prefix, so `train=False` already yields the eval split and there is
    nothing for `imp_prefixes` or `eval_include_vb_ground_truth` to say. This function retires with
    the rest of the h5 path.
    """
    return CandiKitH5Dataset(h5_path, regime, train=False, batch_size=batch_size,
                             biosample_prefix="T_", dsf_sampling="off", seed=seed, shuffle=False,
                             eval_include_vb_ground_truth=True, imp_prefixes=imp_prefixes,
                             cell_cond=eval_cell_cond(model), reference=reference)


def build_eval_units(model, ds, device, *, max_batches: Optional[int] = None,
                     batches_per_pair: Optional[int] = None,
                     meta_probe=None) -> Tuple[List[EvalUnit], List[str]]:
    """Returns (units, assays), scoring whatever eval dataset it is HANDED.

    It takes a ready dataset rather than a path, and that is the whole of t28. Constructing its own
    loader is what pinned this function to the baked h5 and made a store-backed run unscorable; the
    three names it reads off a dataset for the slot arithmetic below — `_eval_indices`,
    `_bios_candidates`, `_all_imp_biosamples` — are spelled identically by `StoreDataset` (t14), so
    with the construction removed the two paths are already interchangeable here.

    `assays` is the dataset's own column order — the only assay labelling the kit ever uses; every
    labelled instrument below takes it as an explicit argument.

    `batch_size` is read off the dataset instead of being a parameter. It only ever fed the cycle
    arithmetic, and a caller that could pass a different number than the loader batches at would
    silently mis-space `batches_per_pair` across the chromosome.
    """
    batch_size = int(ds.batch_size)
    # `batches_per_pair` is the EVEN way to shrink the eval, and the only one used mid-training.
    # `max_batches` truncates the pair cycle and drops whole targets (see the warning below); this
    # keeps EVERY entry and thins the WINDOWS, so the selection metric covers the same target set at
    # every epoch and in every arm. Entries are cycle slots, not (T_, imp) pairs: a T_ cell with no
    # counterpart still occupies one, so the length is counted the way the dataset builds it.
    #
    # Thinning by a PREFIX would be a trap, and not a subtle one. The dataset walks eval windows in
    # genomic order and hands batch `bi` to slot `bi % n_slots`, so the first k*n_slots batches are a
    # contiguous block at the START of the eval chromosome. On chr21 that block is the acrocentric
    # p-arm: measured, all 3,072 positions of every held-out target there are exactly zero, the
    # foreground purity filter drops 100% of records, and the selection metric comes back NaN with
    # n=0 targets. So keep whole CYCLES spaced across the chromosome instead — cycle c covers windows
    # [c*n_slots*bs, (c+1)*n_slots*bs), every slot appears in every cycle, and the midpoint spacing
    # below never starts at cycle 0.
    keep_cycles = None
    n_slots = sum(max(1, len(ds._all_imp_biosamples(t))) for t in ds._bios_candidates())
    if batches_per_pair is not None:
        k = int(batches_per_pair)
        # ASKED OF THE DATASET, never derived from the batch index. The two loaders order eval
        # batches differently -- `CandiKitH5Dataset` interleaves the pairs (batch `bi` belongs to
        # pair `bi % n_pairs`), `StoreDataset` groups them pair-major (all of pair 0's batches, then
        # all of pair 1's). A cycle computed as `bi // n_slots` is correct for the first and, on the
        # second, selects a contiguous block that lies entirely inside the FIRST pair: measured on a
        # two-pair store, `batches_per_pair` scored one target and dropped the other in silence.
        # Thinning WINDOWS must never thin TARGETS -- that is the whole reason this knob exists
        # rather than `max_batches` -- so the cycle is counted per pair, below, off the pair's own
        # name in the batch.
        n_cycles = max(1, int(ds.eval_batches_per_pair()))
        keep_cycles = sorted({min(n_cycles - 1, int((i + 0.5) * n_cycles / k)) for i in range(k)})
    noop = make_masker(p_full_loci=0.0, p_full_assay=0.0, p_chunks=0.0, mask_fraction=0.0)
    model.eval()
    units: List[EvalUnit] = []
    # Each pair's own batch counter. Both loaders emit a pair's batches in genomic order, so a
    # pair's c-th batch is cycle c under either ordering -- which is what makes one counter correct
    # for both without either loader having to change how it iterates.
    pair_cycle: Dict[Tuple[str, str], int] = {}
    last_cycle = max(keep_cycles) if keep_cycles else None
    for bi, batch in enumerate(ds):
        if max_batches is not None and bi >= max_batches:
            break
        if keep_cycles is not None:
            key = (str(batch.get("biosample_name", "")), str(batch.get("imp_biosample_name", "")))
            c = pair_cycle.get(key, 0)
            pair_cycle[key] = c + 1
            if c not in keep_cycles:
                # Stop once every pair has passed its last wanted cycle. On the interleaved order
                # that is the same early exit `max_batches` used to give; on the pair-major order
                # nothing may stop early until the last pair has been reached, and this waits.
                if (len(pair_cycle) >= n_slots
                        and all(v > last_cycle for v in pair_cycle.values())):
                    break
                continue
        prep = prepare_masked_batch(batch, noop, device, apply_mask=False)
        if prep is None:
            continue
        y_meta_fwd = prep["y_meta"].clone()
        ymi = batch.get("y_meta_imp")
        yav = batch.get("y_avail")
        ydi = batch.get("y_data_imp")
        has_imp = isinstance(ymi, torch.Tensor) and isinstance(ydi, torch.Tensor)
        imp_map = ydi_d = ymi_d = yav_d = None
        if has_imp:
            ymi_d = ymi.to(device); yav_d = yav.to(device); ydi_d = ydi.to(device)
            y_meta_fwd = _build_vb_natural_missing_meta(y_meta_fwd, ymi_d, yav_d, None)
            imp_map = ((yav_d <= 0).unsqueeze(1).expand_as(ydi_d)) & (ydi_d != -1)
        # h64: the arm switch, applied here and NOT earlier. `_build_vb_natural_missing_meta` copies
        # the V_/B_ biosample's real covariates into the imp slots, so a row-3 write made before it
        # would be overwritten at exactly the positions the mid-training eval scores. Both prompts and
        # BOTH count targets go through in ONE call, so the encoder, the decoder and the ground truth
        # share one per-sample bin. `record=False`: eval batches must not enter the training arm's
        # own no-op fraction.
        if meta_probe is not None:
            (x_meta_p, y_meta_fwd), (y_data_p, ydi_d) = meta_probe.apply_tensors(
                (prep["x_meta"], y_meta_fwd), (prep["y_data"], ydi_d), record=False)
            prep["x_meta"], prep["y_data"] = x_meta_p, y_data_p
        z = model.encode(prep["x_data"], prep["x_dna"], prep["x_meta"])
        units.append(EvalUnit(
            z=z, y_meta_true=y_meta_fwd, den_map=prep["observed_map"], y_data=prep["y_data"],
            imp_map=imp_map, y_data_imp=ydi_d, y_meta_imp=ymi_d, y_avail=yav_d,
            biosample=str(batch.get("biosample_name", "")),
            imp_biosample=str(batch.get("imp_biosample_name", "")),
            log_ref=prep.get("log_ref")))

    if batches_per_pair is not None:
        seen = {(u.biosample, u.imp_biosample) for u in units if u.imp_biosample}
        print(f"[eval] batches_per_pair={batches_per_pair}: {len(units)} units over {len(seen)} "
              f"of {n_slots} (T_, V_/B_) pairs, {batches_per_pair * batch_size} windows each, from "
              f"cycles {keep_cycles} of {ds.eval_batches_per_pair()} "
              f"(spread across the eval chromosome, never a prefix)", flush=True)
        if len(seen) < n_slots:
            print(f"[eval] WARNING batches_per_pair thinned TARGETS, not just windows: "
                  f"{n_slots - len(seen)} of {n_slots} pairs went unscored. That is a bug in the "
                  f"cycle accounting, not a coverage choice.", flush=True)

    # TRUNCATION IS SILENT AND UNEVEN, so it is named here rather than left to be inferred.
    # The dataset advances one (T_, imp) PAIR per window batch, cycling. So `max_batches=N` over P
    # pairs evaluates only the first N mod-P pairs, and gives each of them ceil-or-floor(N/P) window
    # batches -- not a uniform subsample of every target, but a hard cut that can leave most targets
    # entirely unscored. On the full EIC panel P=38, so a habit like `--eval-max-batches 12` carried
    # over from a 3-pair probe would score 12 targets and drop the other 26 without a word.
    seen_pairs = {(u.biosample, u.imp_biosample) for u in units if u.imp_biosample}
    all_pairs = {(t, i) for t in ds._bios_candidates() for i in ds._all_imp_biosamples(t)}
    if max_batches is not None and len(seen_pairs) < len(all_pairs):
        missed = sorted(all_pairs - seen_pairs)
        print(f"[eval] WARNING max_batches={max_batches} covered {len(seen_pairs)}/{len(all_pairs)} "
              f"(T_, V_/B_) pairs — {len(missed)} targets are UNSCORED, e.g. {missed[:3]}. "
              f"Eval advances one pair per window batch, so max_batches must be >= "
              f"{len(all_pairs)} x (batches wanted per pair). Pass 0/None to score them all.",
              flush=True)
    return units, list(ds.assays)


def decode_latent(model, z: torch.Tensor, y_meta: torch.Tensor,
                  log_ref: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Decode a CACHED encoder latent under (possibly counterfactual) `y_meta`.

    Every measurement here encodes once and re-decodes many times — M2 flips covariates, S14 sweeps
    the prompt depth — so the decode entry point has to be the whole y_meta-conditioned half of the
    model, not just `model.decoder`.

    On the split-conditioning arms (a3/a5) the y_meta attention blocks sit on the DECODER side of the
    boundary. Calling `model.decoder(z, ...)` directly would silently skip them: the run would score,
    the numbers would look plausible, and a3 would be measured as a2 with extra untrained weights.
    Arms that expose `decode_latent` get it; candi_kit's shipped model has no such method and falls
    through to the historical call, unchanged.
    """
    fn = getattr(model, "decode_latent", None)
    return fn(z, y_meta, log_ref) if fn is not None else model.decoder(z, y_meta, log_ref)


@torch.no_grad()
def _decode(model, u: EvalUnit, y_meta: torch.Tensor) -> Dict[str, torch.Tensor]:
    return decode_latent(model, u.z, y_meta, u.log_ref)


def _gather(out: Dict[str, torch.Tensor], mask: torch.Tensor, a: int,
            target: torch.Tensor) -> Optional[Dict[str, np.ndarray]]:
    """Pool assay a's per-position (mu,n,p,eta,log2_mu,target) over a [B,L,A] bool mask; p from true mu."""
    m = mask[:, :, a]
    if not bool(m.any()):
        return None
    mu = out["mu"][:, :, a][m].double().cpu().numpy()
    n = out["n"][:, :, a][m].double().cpu().numpy()
    eta = out["eta"][:, :, a][m].float().cpu().numpy()
    log2_mu = out["log2_mu"][:, :, a][m].double().cpu().numpy()
    tgt = target[:, :, a][m].double().cpu().numpy()
    return dict(mu=mu, n=n, p=_p_from_true_mu(n, mu), eta=eta, log2_mu=log2_mu, target=tgt)


# ---------------------------------------------------------------------------
# honest per-assay marginal + oracle scale decomposition
# ---------------------------------------------------------------------------

def _marginal_nb(target: np.ndarray) -> Dict:
    """The CRPS-OPTIMAL constant NB forecast for one assay (the honest bar).

    The old baseline used `mu = median(target) + 1e-6`. On 4/8 assays the median count is 0, so the
    "NB marginal" was a point mass at 0 and `marg_crps` reduced to `mean(y)` -- a forecast of
    "predict nothing" that any model beats, which is how "beats marginal 8/8" was manufactured.

    Since the forecast is constant, its CRPS depends on the target only through its HISTOGRAM, so the
    (mu, n) grid search is exact and cheap on the unique values.
    """
    if target.size < 2:
        return dict(marg_crps=float("nan"), marg_mu=float("nan"), marg_n=float("nan"),
                    marg_mu_legacy_median=float("nan"), marg_crps_legacy_median=float("nan"))
    vals, cnts = np.unique(target, return_counts=True)
    w = cnts / cnts.sum()
    mean = float(np.dot(w, vals))
    var = float(np.dot(w, (vals - mean) ** 2))
    mu0 = max(mean, 1e-3)
    n0 = max((mu0 * mu0) / max(var - mu0, 1e-3), 1e-3)                  # NB var = mu + mu^2/n

    def _crps_const(mu_c, n_c):
        nv, mv = np.full_like(vals, n_c), np.full_like(vals, mu_c)
        return float(np.dot(w, nb_crps(nv, _p_from_true_mu(nv, mv), vals)))

    # the legacy median-based baseline, kept so the degeneracy is visible rather than asserted -- and
    # entered as a CANDIDATE, so the honest bar provably dominates it (on an all-zero target a near
    # point mass genuinely IS the CRPS-optimal constant; the audit's objection was that the MEDIAN rule
    # produced one when it was not optimal, not that it can never be).
    med = float(np.median(target)) + 1e-6
    med_n = max((med * med) / max(var - med, 1e-6), 1e-6)
    crps_legacy = _crps_const(med, med_n)
    best = (crps_legacy, med, med_n)
    for c in np.arange(-8.0, 3.01, 0.25):
        mu_c = mu0 * 2.0 ** c
        for f in np.arange(-3.0, 3.01, 0.5):
            n_c = max(n0 * 2.0 ** f, 1e-6)
            v = _crps_const(mu_c, n_c)
            if v < best[0]:
                best = (v, mu_c, n_c)
    return dict(marg_crps=best[0], marg_mu=best[1], marg_n=best[2],
                marg_mu_legacy_median=med, marg_crps_legacy_median=crps_legacy)


def _oracle_scale(mu: np.ndarray, n: np.ndarray, target: np.ndarray, *,
                  fit_budget: int = 20_000, seed: int = 0) -> Dict:
    """Oracle per-assay multiplicative scale `c* = argmin_c CRPS(NB(n, mu*2^c), y)`.

    Macro CRPS is location-dominated (a 4x mu error costs +84%, a 4x dispersion error +9%), so an arm
    that wins on scale and an arm that wins on shape sum into an apparent Pareto. This splits it:
    `crps_oracle_scaled` is the CAPABILITY term (what the model gets right once its per-assay level is
    granted) and `scale_error = crps - crps_oracle_scaled` is the FIXABLE calibration term.
    `n_star_log2` is the analogous oracle dispersion rescale.

    The grid search runs on a subsample; the reported CRPS values are evaluated on the FULL pool at the
    selected oracles, so they are directly comparable to the un-decomposed `crps`. That makes
    `crps_oracle_scaled` an IN-SAMPLE oracle -- an upper bound on capability -- and lets `scale_error`
    go slightly negative (-0.0008 observed).
    """
    N = len(mu)
    sel = (np.arange(N) if N <= fit_budget
           else np.random.default_rng(seed).choice(N, fit_budget, replace=False))
    mf, nf, tf = mu[sel], n[sel], target[sel]

    def _fit(c, k=0.0):
        m, nv = mf * 2.0 ** c, nf * 2.0 ** k
        return float(np.mean(nb_crps(nv, _p_from_true_mu(nv, m), tf)))

    c = float(min(np.arange(-6.0, 6.001, 0.25), key=_fit))
    c = float(min(np.arange(c - 0.25, c + 0.2501, 0.01), key=_fit))
    k = float(min(np.arange(-4.0, 4.001, 0.25), key=lambda kk: _fit(c, kk)))
    ms = mu * 2.0 ** c
    nk = n * 2.0 ** k
    return dict(c_star=c, n_star_log2=k,
                crps_oracle_scaled=float(np.mean(nb_crps(n, _p_from_true_mu(n, ms), target))),
                crps_oracle_scaled_and_n=float(np.mean(nb_crps(nk, _p_from_true_mu(nk, ms), target))))


# ---------------------------------------------------------------------------
# M1 — counts-only reconstruction / imputation health
# ---------------------------------------------------------------------------

@torch.no_grad()
def _spearman_diag(mu, tgt, label: str) -> float:
    """spearman(), but says WHY it is nan instead of emitting a bare nan.

    Undefined rank correlation is a real outcome on a small panel -- a held-out target that is all-zero
    on the eval chromosome has no ranking to recover. Reported silently it reads as a broken metric, so
    name the cause at the point it happens.
    """
    v = spearman(mu, tgt)
    if not np.isfinite(v):
        import sys as _sys
        why = ("target is constant (%d points, all == %g)" % (len(tgt), tgt[0])
               if len(tgt) and float(np.std(tgt)) < 1e-12 else
               "prediction is constant" if len(mu) and float(np.std(mu)) < 1e-12 else
               "fewer than 2 points (n=%d)" % len(tgt))
        print("[eval] WARNING %s: Spearman undefined -- %s. This is a property of the eval split, "
              "not a model failure; widen the panel or the eval chromosome set." % (label, why),
              file=_sys.stderr)
    return v


def eval_M1(model, units: List[EvalUnit], device, assays: Sequence[str], *,
            budget: int = 200_000, seed: int = 0, include_deprecated: bool = False) -> Dict:
    # `include_deprecated` is accepted for a uniform evaluate() call; M1 emits no deprecated keys.
    den, imp, zs = [], [], []
    A = len(assays)
    den_a = {a: [] for a in range(A)}
    imp_a = {a: [] for a in range(A)}
    # Per-TARGET accumulation, keyed (T_biosample, imp_biosample, assay). The per-assay pools above
    # concatenate across every eval unit, which discards which CELL each point came from — and the
    # cell is the replication unit for any arm-vs-arm comparison (see `_cluster_bootstrap_ci`: the
    # effective n is targets, not positions). Several units share one (T_, imp) pair because the eval
    # cycles pairs across window batches, so records accumulate per key rather than overwrite.
    den_t: Dict[Tuple[str, str, str], List[Dict]] = {}
    imp_t: Dict[Tuple[str, str, str], List[Dict]] = {}
    for u in units:
        out = _decode(model, u, u.y_meta_true)
        for a in range(A):
            g = _gather(out, u.den_map, a, u.y_data)
            if g is not None:
                den.append(g)
                den_a[a].append(g)
                den_t.setdefault((u.biosample, u.biosample, assays[a]), []).append(g)
        if u.imp_map is not None:
            for a in u.imp_target_assays():
                g = _gather(out, u.imp_map, a, u.y_data_imp)
                if g is not None:
                    imp.append(g)
                    imp_a[a].append(g)
                    imp_t.setdefault((u.biosample, u.imp_biosample, assays[a]), []).append(g)
        zs.append(u.z.reshape(-1, u.z.shape[-1]).float().cpu().numpy())   # [B*L2, d] per-position latent

    def _pool(parts):
        if not parts:
            return None
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def _suite(pool, tag):
        if pool is None:
            return dict(n_points=0)
        rng = np.random.default_rng(seed)
        n0 = len(pool["mu"])
        sel = np.arange(n0) if n0 <= budget else rng.choice(n0, budget, replace=False)
        mu, n, p, tgt = pool["mu"][sel], pool["n"][sel], pool["p"][sel], pool["target"][sel]
        grid, fbar = calibration_pit_curve(n, p, tgt)
        # spearman is on RAW counts, pearson on log1p -- two different spaces under one heading, so the
        # keys carry their space and can never be quoted as a pair.
        return dict(spearman_raw=_spearman_diag(mu, tgt, tag), pearson_log1p=pearson(np.log1p(mu), np.log1p(tgt)),
                    crps=float(np.mean(nb_crps(n, p, tgt))), ece=ece(n, p, tgt),
                    r2=r2(mu, tgt), n_points=int(len(mu)), calib_grid=grid, calib_fbar=fbar)

    den_pool, imp_pool = _pool(den), _pool(imp)
    m_den, m_imp = _suite(den_pool, "den"), _suite(imp_pool, "imp")

    # Per-assay decomposition. Pooled Spearman/Pearson mix assays of very different count magnitudes,
    # so they reward getting the CROSS-ASSAY SCALE right -- exactly the offset's free 2^(d-center)
    # arithmetic. Per-assay corr is scale-free (within-assay ranking) => isolates SHAPE (the biology);
    # median_mu vs median_target exposes the cross-assay scale bias directly.
    def _suite_light(pool):
        n0 = len(pool["mu"])
        sel = np.arange(n0) if n0 <= budget else np.random.default_rng(seed).choice(n0, budget, replace=False)
        mu, n, p, tgt = pool["mu"][sel], pool["n"][sel], pool["p"][sel], pool["target"][sel]
        crps = float(np.mean(nb_crps(n, p, tgt)))
        mar = _marginal_nb(tgt)                                         # CRPS-optimal, not median
        orc = _oracle_scale(mu, n, tgt, seed=seed)                      # scale / capability split
        mcrps = mar["marg_crps"]
        return dict(spearman_raw=spearman(mu, tgt), pearson_log1p=pearson(np.log1p(mu), np.log1p(tgt)),
                    crps=crps, r2=r2(mu, tgt),
                    mse_log=float(np.mean((np.log1p(mu) - np.log1p(tgt)) ** 2)),
                    marg_crps=mcrps, beats_marginal=bool(np.isfinite(mcrps) and crps < mcrps),
                    marg_mu=mar["marg_mu"], marg_n=mar["marg_n"],
                    marg_mu_legacy_median=mar["marg_mu_legacy_median"],
                    marg_crps_legacy_median=mar["marg_crps_legacy_median"],
                    **orc, scale_error=crps - orc["crps_oracle_scaled"],
                    beats_marginal_oracle_scaled=bool(np.isfinite(mcrps)
                                                      and orc["crps_oracle_scaled"] < mcrps),
                    median_mu=float(np.median(mu)), median_target=float(np.median(tgt)),
                    n_points=int(len(mu)))

    def _per_assay(by_assay):
        return {assays[a]: _suite_light(_pool(parts))
                for a, parts in by_assay.items() if parts}

    def _macro(pa, key):
        vals = [v[key] for v in pa.values() if np.isfinite(v.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    def _per_target(by_target):
        # "T_x|V_x|assay" — a flat string key so it survives the JSON round-trip that
        # `candi.compare_arms` reads back. Split on '|' to recover the tuple.
        return {"|".join(k): _suite_light(_pool(parts))
                for k, parts in sorted(by_target.items()) if parts}

    imp_pa, den_pa = _per_assay(imp_a), _per_assay(den_a)
    imp_pt, den_pt = _per_target(imp_t), _per_target(den_t)
    Z = np.concatenate(zs) if zs else np.zeros((0, 1))
    if len(Z) > budget:
        Z = Z[np.random.default_rng(seed).choice(len(Z), budget, replace=False)]
    eff = effective_rank(Z)
    return dict(imp=m_imp, den=m_den, encoder_eff_rank_perpos=eff,
                imp_per_assay=imp_pa, den_per_assay=den_pa,
                imp_per_target=imp_pt, den_per_target=den_pt,
                imp_macro_spearman_raw=_macro(imp_pa, "spearman_raw"),
                imp_macro_pearson_log1p=_macro(imp_pa, "pearson_log1p"),
                den_macro_spearman_raw=_macro(den_pa, "spearman_raw"),
                den_macro_pearson_log1p=_macro(den_pa, "pearson_log1p"),
                # macro CRPS split into the capability term and the fixable per-assay scale term.
                imp_macro_crps=_macro(imp_pa, "crps"),
                imp_macro_crps_oracle_scaled=_macro(imp_pa, "crps_oracle_scaled"),
                imp_macro_scale_error=_macro(imp_pa, "scale_error"),
                imp_macro_marg_crps=_macro(imp_pa, "marg_crps"),
                imp_beats_marginal_n=int(sum(v.get("beats_marginal", False) for v in imp_pa.values())),
                imp_beats_marginal_oracle_scaled_n=int(
                    sum(v.get("beats_marginal_oracle_scaled", False) for v in imp_pa.values())),
                den_macro_crps=_macro(den_pa, "crps"))


# ---------------------------------------------------------------------------
# M2 — counterfactual-prompt FLIP (depth, run_type, read_length)
# ---------------------------------------------------------------------------

def _flip_run_type(y_meta: torch.Tensor, a: int) -> torch.Tensor:
    ym = y_meta.clone()
    rt = ym[:, RUN_TYPE_ROW, a]
    ym[:, RUN_TYPE_ROW, a] = torch.where((rt == 0) | (rt == 1), 1.0 - rt, rt)   # 0<->1; leave sentinels
    return ym


def _flip_read_length(y_meta: torch.Tensor, a: int) -> torch.Tensor:
    ym = y_meta.clone()
    obs = torch.tensor(OBSERVED_READLENS, device=y_meta.device)
    cur = ym[:, READLEN_ROW, a]
    for b in range(ym.shape[0]):
        v = float(cur[b])
        if v <= 0:
            continue
        far = obs[(obs - v).abs().argmax()]    # flip to the FARTHEST observed read length
        ym[b, READLEN_ROW, a] = far
    return ym


def _set_depth(y_meta: torch.Tensor, a: int, depth_vec: torch.Tensor) -> torch.Tensor:
    ym = y_meta.clone()
    ym[:, DEPTH_ROW, a] = depth_vec
    return ym


def _target_run_type_label(u: "EvalUnit", a: int) -> str:
    """TRUE run_type of held-out target assay a, read off the VB-natural prompt (row 3)."""
    if u.y_meta_imp is None:
        return "?"
    rt = float(u.y_meta_imp[0, RUN_TYPE_ROW, a])
    return "paired" if rt == 1.0 else ("single" if rt == 0.0 else "?")


@torch.no_grad()
def _flip_covariate(model, units, device, covariate: str, assays: Sequence[str], *, fg_frac: float,
                    n_boot: int, seed: int, include_deprecated: bool = False) -> Dict:
    """Direction + responsiveness for a categorical flip (run_type / read_length) over held-out targets."""
    per_target, pooled_delta, split = [], {"all": []}, {"single": [], "paired": []}
    for u in units:
        if u.imp_map is None:
            continue
        for a in u.imp_target_assays():
            ym_true = u.y_meta_true
            ym_flip = _flip_run_type(ym_true, a) if covariate == "run_type" else _flip_read_length(ym_true, a)
            g_true = _gather(_decode(model, u, ym_true), u.imp_map, a, u.y_data_imp)
            g_flip = _gather(_decode(model, u, ym_flip), u.imp_map, a, u.y_data_imp)
            if g_true is None or g_flip is None:
                continue
            fg, fired = _foreground_mask(g_true["target"], fg_frac, seed=seed)
            crps_true = nb_crps(g_true["n"][fg], g_true["p"][fg], g_true["target"][fg])
            crps_flip = nb_crps(g_flip["n"][fg], g_flip["p"][fg], g_flip["target"][fg])
            delta = crps_flip - crps_true                     # >0 means TRUE prompt is better (direction)
            resp = float(np.mean(np.abs(g_true["mu"][fg] - g_flip["mu"][fg])))
            key = (u.biosample, u.imp_biosample, assays[a])
            label = _target_run_type_label(u, a)
            rec = dict(target=key, run_type=label, n_fg=int(fg.sum()),
                       fg_frac_realized=float(fg.mean()),
                       purity_fallback_fired=bool(fired), fg_pos_mean=_fg_pos_mean(fg),
                       crps_true=float(np.mean(crps_true)), crps_flip=float(np.mean(crps_flip)),
                       mean_delta=float(np.mean(delta)), responsiveness=resp,
                       direction_true_better=bool(np.mean(delta) > 0))
            per_target.append(rec)
            pooled_delta["all"].append(delta)
            if label in ("single", "paired"):
                split[label].append(delta)

    def _agg(dlist):
        if not dlist:
            return dict(n=0)
        return _bootstrap_ci(np.concatenate(dlist), n_boot=n_boot, seed=seed)

    def _cl(recs):
        return _cluster_bootstrap_ci(recs, n_boot=n_boot, seed=seed)

    by_label = {lab: [r for r in per_target if r["run_type"] == lab] for lab in ("single", "paired")}
    out = dict(covariate=covariate, per_target=per_target,
               # PRIMARY: target-clustered, n_fg-weighted, sign-aware.
               overall_clustered=_cl(per_target), single_clustered=_cl(by_label["single"]),
               paired_clustered=_cl(by_label["paired"]),
               n_targets=len(per_target),
               mean_responsiveness=float(np.mean([r["responsiveness"] for r in per_target]))
               if per_target else float("nan"))
    # honest-null flag: a bit-exactly zero prompt response is a MODEL statistic, so it is named as one.
    out["model_unresponsive"] = bool(
        per_target and out["mean_responsiveness"] < 1e-6)
    if include_deprecated:
        out.update(overall=_agg(pooled_delta["all"]),
                   single=_agg(split["single"]), paired=_agg(split["paired"]),
                   frac_direction=float(np.mean([r["direction_true_better"] for r in per_target]))
                   if per_target else float("nan"),
                   deprecated_verdicts={k: DEPRECATED_VERDICTS[k]
                                        for k in ("overall", "single", "paired", "frac_direction")})
    return out


@torch.no_grad()
def _depth_sweep(model, units, device, assays: Sequence[str], *, fg_frac: float, n_boot: int, seed: int,
                 include_deprecated: bool = False) -> Dict:
    """Depth: sweep told-depth over the imp biosample's achievable meta_dsf levels; direction +
    offset-independent eta/tail regression + shuffled-prompt null."""
    per_target = []
    pooled_dir_delta, pooled_null_delta = [], []
    dec = getattr(model, "decoder", None)
    src = dec if hasattr(dec, "clamp_lo") else model     # real model keeps it on .decoder; stubs on self
    clamp_lo = float(getattr(src, "clamp_lo", -np.inf))
    clamp_hi = float(getattr(src, "clamp_hi", np.inf))
    for u in units:
        if u.imp_map is None or u.y_meta_imp is None:
            continue
        for a in u.imp_target_assays():
            base_d = u.y_meta_imp[:, DEPTH_ROW, a]              # [B] true (VB-natural) depth
            if not torch.isfinite(base_d).all() or bool((base_d < 0).any()):
                continue
            told = [base_d - float(np.log2(k)) for k in (1, 2, 4, 8)]   # achievable set (downsample)
            gs, eta_means, tail_means, crps_means, dvals = [], [], [], [], []
            log2mu_means, sat_fracs = [], []
            fired_any = False
            for dvec in told:
                g = _gather(_decode(model, u, _set_depth(u.y_meta_true, a, dvec)), u.imp_map, a, u.y_data_imp)
                if g is None:
                    gs = []
                    break
                fg, fired = _foreground_mask(g["target"], fg_frac, seed=seed)
                fired_any = fired_any or fired
                gs.append((g, fg))
                eta_means.append(float(np.mean(g["eta"][fg])))
                log2mu_means.append(float(np.mean(g["log2_mu"][fg])))
                lm = g["log2_mu"][fg]
                sat_fracs.append(float(np.mean((lm <= clamp_lo + 1e-6) | (lm >= clamp_hi - 1e-6))))
                tail_means.append(float(np.mean(nb_quantile(0.95, g["n"][fg], g["p"][fg]))))
                crps_means.append(float(np.mean(nb_crps(g["n"][fg], g["p"][fg], g["target"][fg]))))
                dvals.append(float(dvec.float().mean()))
            if not gs:
                continue
            # direction: true depth (k=1, index 0) should have the lowest CRPS vs GT
            g_true, fg_true = gs[0]
            crps_true = nb_crps(g_true["n"][fg_true], g_true["p"][fg_true], g_true["target"][fg_true])
            # wrong depth = the most-downsampled told (k=8, index -1)
            g_wrong, fg_w = gs[-1]
            crps_wrong = nb_crps(g_wrong["n"][fg_true] if fg_w.shape == fg_true.shape else g_wrong["n"][fg_w],
                                 g_wrong["p"][fg_true] if fg_w.shape == fg_true.shape else g_wrong["p"][fg_w],
                                 g_true["target"][fg_true] if fg_w.shape == fg_true.shape else g_wrong["target"][fg_w])
            dir_delta = crps_wrong - crps_true
            # PRIMARY: total told-depth response d<log2_mu>/d(told depth); a correct exposure model
            # gives 1.0. `eta_slope` below is the offset-FREE residual = 1 - (offset coefficient), i.e. it
            # measures WHERE the response is implemented, not WHETHER the model responds -- demoted to a
            # labelled attribution diagnostic (`total_slope = beta + eta_slope` for any hybrid).
            dv = np.array(dvals)
            total_slope = float(np.polyfit(dv, np.array(log2mu_means), 1)[0]) if np.ptp(dv) > 1e-6 else float("nan")
            eta_slope = float(np.polyfit(dv, np.array(eta_means), 1)[0]) if np.ptp(dv) > 1e-6 else float("nan")
            tail_slope = float(np.polyfit(dv, np.log2(np.array(tail_means) + 1.0), 1)[0]) if np.ptp(dv) > 1e-6 else float("nan")
            # null: told a SHUFFLED (permuted-across-samples) depth -> direction should collapse
            perm = torch.randperm(base_d.shape[0],
                                  generator=torch.Generator().manual_seed(seed)).to(base_d.device)
            g_shuf = _gather(_decode(model, u, _set_depth(u.y_meta_true, a, base_d[perm])),
                             u.imp_map, a, u.y_data_imp)
            null_delta = np.zeros(0)
            if g_shuf is not None:
                null_delta = (nb_crps(g_shuf["n"][fg_true], g_shuf["p"][fg_true], g_true["target"][fg_true])
                              - crps_true)
            per_target.append(dict(
                target=(u.biosample, u.imp_biosample, assays[a]),
                crps_curve=crps_means, told_depth=dvals, eta_means=eta_means, tail_means=tail_means,
                log2mu_means=log2mu_means, n_fg=int(fg_true.sum()), fg_frac_realized=float(fg_true.mean()),
                purity_fallback_fired=bool(fired_any), fg_pos_mean=_fg_pos_mean(fg_true),
                frac_log2mu_at_clamp=float(np.max(sat_fracs)),
                min_at_true=bool(int(np.argmin(crps_means)) == 0),
                total_slope=total_slope, eta_slope=eta_slope, tail_slope=tail_slope,
                dir_mean_delta=float(np.mean(dir_delta)),
                null_mean_delta=float(np.mean(null_delta)) if null_delta.size else float("nan")))
            pooled_dir_delta.append(dir_delta)
            if null_delta.size:
                pooled_null_delta.append(null_delta)
    tot_slopes = [t["total_slope"] for t in per_target if np.isfinite(t["total_slope"])]
    med_tot = float(np.median(tot_slopes)) if tot_slopes else float("nan")
    out = dict(covariate="depth", per_target=per_target,
               # PRIMARY: target-clustered.
               direction_clustered=_cluster_bootstrap_ci(per_target, value_key="dir_mean_delta",
                                                         n_boot=n_boot, seed=seed),
               n_targets=len(per_target),
               # PRIMARY: total told-depth response; |median_total_slope - 1| is the honest bar.
               median_total_slope=med_tot, total_slope_err=abs(med_tot - 1.0),
               # ...but a slope measured through the saturating `log2_mu` clamp is not an exposure
               # coefficient. Flag it, or a saturated arm reads as sub-unit response the way `eta_slope`
               # read as "no response". The MEDIAN is 0 on all four recorded arms at full coverage, but
               # the distribution is heavy-tailed -- `wd0_on_s0` has 205/1215 targets with a nonzero
               # clamp fraction (mean 0.096, p90 0.475, max 0.967) -- so read `total_slope` next to the
               # tail stats, not the median alone.
               median_frac_log2mu_at_clamp=float(np.median([t["frac_log2mu_at_clamp"] for t in per_target]))
               if per_target else float("nan"),
               frac_targets_any_clamp=float(np.mean([t["frac_log2mu_at_clamp"] > 0 for t in per_target]))
               if per_target else float("nan"),
               p90_frac_log2mu_at_clamp=float(np.quantile([t["frac_log2mu_at_clamp"] for t in per_target],
                                                          0.9)) if per_target else float("nan"),
               max_frac_log2mu_at_clamp=float(np.max([t["frac_log2mu_at_clamp"] for t in per_target]))
               if per_target else float("nan"),
               total_slope_clamp_saturated=bool(
                   per_target and np.median([t["frac_log2mu_at_clamp"] for t in per_target]) > 0.01))
    if include_deprecated:
        eta_slopes = [t["eta_slope"] for t in per_target if np.isfinite(t["eta_slope"])]
        out.update(
            null_clustered=_cluster_bootstrap_ci(
                [t for t in per_target if np.isfinite(t["null_mean_delta"])],
                value_key="null_mean_delta", n_boot=n_boot, seed=seed),
            direction=_bootstrap_ci(np.concatenate(pooled_dir_delta), n_boot=n_boot, seed=seed)
            if pooled_dir_delta else dict(n=0),
            null=_bootstrap_ci(np.concatenate(pooled_null_delta), n_boot=n_boot, seed=seed)
            if pooled_null_delta else dict(n=0),
            frac_min_at_true=float(np.mean([t["min_at_true"] for t in per_target]))
            if per_target else float("nan"),
            median_eta_slope=float(np.median(eta_slopes)) if eta_slopes else float("nan"),
            eta_slope_is_attribution_diagnostic=True,
            offset_independent=bool(eta_slopes and np.median(eta_slopes) > 0.0),
            deprecated_verdicts={k: DEPRECATED_VERDICTS[k] for k in
                                 ("null", "null_clustered", "direction", "frac_min_at_true",
                                  "median_eta_slope", "offset_independent")})
    return out


META_ROWS = {0: "depth", 1: "assay_id", 2: "read_length", 3: "run_type"}


@torch.no_grad()
def _metadata_ablation(model, units, row: int, assays: Sequence[str], mode: str = "cross_target", *,
                       fg_frac: float = 0.02, seed: int = 0) -> Dict:
    """Covariate-agnostic "does the decoder use `y_meta` row `row` AT ALL?" probe (real z).

    For every held-out imputation target, re-decode the CACHED encoder `z` with ONE prompt row replaced:
      - `cross_target` (primary): by the value of a DIFFERENT eval target. Required because `meta_dsf{k}`
        has no window axis, so a *within-batch* permutation is structurally the identity and would report
        "uses nothing" for every model.
      - `within_batch`: that degenerate control, kept so the claim is testable rather than asserted; it
        must read exactly 0.
    Reports dCRPS (>0 = the honest prompt is better), mean/max |d mu| and |d eta| (offset-free), so a dead
    embedder reads bit-exactly 0 on all three regardless of what the offset arithmetic does.

    SENTINEL-FREE by construction: both the true value and the donor value are drawn from held-out
    targets, so both are REAL covariate values. A whole-row permutation instead moves the MISSING/CLOZE
    sentinel onto (and off) unavailable slots, and the resulting response measures "is this slot
    present?" -- not "which assay is this?" -- on an off-manifold prompt. That contamination is large
    enough to dominate the statistic: it is what made a 0.833 assay-steering reading 99.95% artifact
    (sentinel-free: 0.0023).
    """
    targets, n_sentinel_skipped = [], 0
    for ui, u in enumerate(units):
        if u.imp_map is None:
            continue
        for a in u.imp_target_assays():
            v = u.y_meta_true[:, row, a]
            if not torch.isfinite(v).all() or bool(((v == MISSING) | (v == CLOZE)).any()):
                n_sentinel_skipped += 1
                continue
            targets.append((ui, a, float(v[0])))
    per_target = []
    N = len(targets)
    for i, (ui, a, val) in enumerate(targets):
        u = units[ui]
        ym_true = u.y_meta_true
        ym_swap = ym_true.clone()
        if mode == "cross_target":
            donor = next((targets[(i + j) % N] for j in range(1, N) if abs(targets[(i + j) % N][2] - val) > 1e-9), None)
            if donor is None:
                continue
            ym_swap[:, row, a] = float(donor[2])
            swapped_value = float(donor[2])
        elif mode == "within_batch":
            perm = torch.randperm(ym_true.shape[0],
                                  generator=torch.Generator().manual_seed(seed + i)).to(ym_true.device)
            ym_swap[:, row, a] = ym_true[perm, row, a]
            swapped_value = float(ym_swap[0, row, a])
        else:
            raise ValueError(mode)
        g_t = _gather(_decode(model, u, ym_true), u.imp_map, a, u.y_data_imp)
        g_s = _gather(_decode(model, u, ym_swap), u.imp_map, a, u.y_data_imp)
        if g_t is None or g_s is None:
            continue
        fg, fired = _foreground_mask(g_t["target"], fg_frac, seed=seed)
        crps_t = float(np.mean(nb_crps(g_t["n"][fg], g_t["p"][fg], g_t["target"][fg])))
        crps_s = float(np.mean(nb_crps(g_s["n"][fg], g_s["p"][fg], g_s["target"][fg])))
        d_eta = np.abs(g_s["eta"][fg] - g_t["eta"][fg])
        per_target.append(dict(
            target=(u.biosample, u.imp_biosample, assays[a]), n_fg=int(fg.sum()),
            fg_frac_realized=float(fg.mean()), true_value=val, swapped_value=swapped_value,
            purity_fallback_fired=bool(fired), fg_pos_mean=_fg_pos_mean(fg),
            d_crps=crps_s - crps_t,
            d_mu=float(np.mean(np.abs(g_s["mu"][fg] - g_t["mu"][fg]))),
            d_eta=float(np.mean(d_eta)), d_eta_max=float(np.max(d_eta))))

    def _m(k):
        return float(np.mean([t[k] for t in per_target])) if per_target else float("nan")

    return dict(row=int(row), covariate=META_ROWS.get(int(row), str(row)), mode=mode,
                per_target=per_target, n_targets=len(per_target),
                n_sentinel_skipped=n_sentinel_skipped,
                d_crps_clustered=_cluster_bootstrap_ci(per_target, value_key="d_crps", seed=seed),
                mean_d_crps=_m("d_crps"), mean_abs_d_mu=_m("d_mu"), mean_abs_d_eta=_m("d_eta"),
                # max is the statistic the V2 bar was written against ("permuting assay_id gives
                # max|d_eta| >= 0.10"), reported here on a sentinel-free swap so the two are comparable.
                max_abs_d_eta=float(np.max([t["d_eta_max"] for t in per_target])) if per_target
                else float("nan"),
                frac_true_better=float(np.mean([t["d_crps"] > 0 for t in per_target])) if per_target
                else float("nan"),
                uses_covariate=bool(per_target and _m("d_eta") > 0.0))


@torch.no_grad()
def eval_M2(model, units, device, assays: Sequence[str], *, fg_frac: float = 0.02, n_boot: int = 1000,
            seed: int = 0, include_deprecated: bool = False) -> Dict:
    out = dict(
        run_type=_flip_covariate(model, units, device, "run_type", assays, fg_frac=fg_frac,
                                 n_boot=n_boot, seed=seed, include_deprecated=include_deprecated),
        depth=_depth_sweep(model, units, device, assays, fg_frac=fg_frac, n_boot=n_boot, seed=seed,
                           include_deprecated=include_deprecated),
        ablation={META_ROWS[r]: _metadata_ablation(model, units, r, assays, fg_frac=fg_frac, seed=seed)
                  for r in (0, 1, 2, 3)},
        # the structural null: within_batch must read exactly 0 on every arm.
        ablation_within_batch={META_ROWS[r]: _metadata_ablation(model, units, r, assays,
                                                                mode="within_batch",
                                                                fg_frac=fg_frac, seed=seed)
                               for r in (0, 1, 2, 3)})
    if include_deprecated:
        out["read_length"] = _flip_covariate(model, units, device, "read_length", assays,
                                             fg_frac=fg_frac, n_boot=n_boot, seed=seed,
                                             include_deprecated=include_deprecated)
        out["deprecated_verdicts"] = {"read_length": DEPRECATED_VERDICTS["read_length"]}
    return out


# ---------------------------------------------------------------------------
# M3 — shared-biological-latent invariance across input DSF conditions
# ---------------------------------------------------------------------------

def _open_h5(h5_path):
    p = Path(h5_path)
    return h5py.File(p, "r")


def _window_pool(h, chrom: Optional[Sequence[str]]) -> np.ndarray:
    """Window indices on the eval chromosomes. `chrom=None` -> the h5's own `eval_chroms` attr, so the
    eval region is a property of the baked file rather than a hardcoded default."""
    sel = list(chrom) if chrom is not None else list(json.loads(h.attrs["eval_chroms"]))
    chroms = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in h["windows/chrom"][:]])
    return np.where(np.isin(chroms, sel))[0]


def _model_num_cells(model) -> int:
    """Size of the model's cell-identity table; 0 when the 5th metadata row is off."""
    return int(getattr(model.encoder.metadata_embedding, "num_cells", 0))


_CELL_ID_CACHE: Dict[str, Dict[str, int]] = {}


def _h5_cell_ids(h5) -> Dict[str, int]:
    """Memoised per h5 file. `_dsf_counterfactual` calls this from inside four nested loops
    (target x region draw x dsf level x told level), and re-parsing the biosample-order JSON on
    every one of them is pure waste."""
    key = str(getattr(h5, "filename", id(h5)))
    if key not in _CELL_ID_CACHE:
        _CELL_ID_CACHE[key] = cell_id_map(json.loads(h5["biosamples"].attrs["order"]))
    return _CELL_ID_CACHE[key]


def _with_cell_row(meta: torch.Tensor, gname: str, h5, model) -> torch.Tensor:
    """Append metadata row 4 (cell identity) to a [B, 4, F] tensor built straight off the h5.

    M3 and the S14 depth counterfactual assemble their own prompts rather than going through the
    dataset, so they need the row added here or they hand a 4-row tensor to a 5-row embedder.

    The id is always the REAL cell type, in every arm. Under `cell_cond="random"` the model was
    trained to treat row 4 as noise, so which valid id these deterministic instruments feed cannot
    change what they measure -- and if it somehow does, that is itself worth seeing rather than
    hiding behind a second RNG stream.
    """
    if _model_num_cells(model) <= 0:
        return meta
    cid = float(_h5_cell_ids(h5)[base_cell_type(gname)])
    pad = torch.full((meta.shape[0], 1, meta.shape[2]), cid,
                     dtype=meta.dtype, device=meta.device)
    return torch.cat([meta, pad], dim=1)


@torch.no_grad()
def _encode_region_at_dsf(model, h5, gname, wi, dsf, device, *, pool: bool = True) -> np.ndarray:
    """Encode a (biosample, windows) region using counts_dsf{dsf} + meta_dsf{dsf} (+control,dna).
    All available assays present with their TRUE metadata -> Z = mean-over-length latent [B, d]
    (`pool=False` returns the full [B, L2, d] latent, which the depth counterfactual decodes from)."""
    g = h5["biosamples"][gname]
    B = len(wi)
    L = g["pval"].shape[1]
    md = np.array(g[f"meta_dsf{dsf}"])                      # [4, F]
    F = md.shape[1]
    x_data = torch.full((B, L, F), -1.0)
    x_meta = torch.full((B, 4, F), -1.0)
    for fi in range(F):
        if float(md[0, fi]) != -1.0:
            x_meta[:, :, fi] = torch.tensor(md[:, fi], dtype=torch.float32)
            cnt = np.array(g[f"counts_dsf{dsf}"][np.sort(wi), :, fi])
            x_data[:, :, fi] = torch.tensor(cnt, dtype=torch.float32)
    control = torch.tensor(np.array(g["control"][np.sort(wi)]), dtype=torch.float32)      # [B,L,1]
    cmeta = torch.tensor(np.array(g["control_meta"][np.sort(wi)]), dtype=torch.float32)   # [B,4,1]
    # mark control missing where the h5 stores -1 (encoder asserts signal/meta availability agree)
    ctrl_missing = (control[:, :, 0] == -1.0).any(dim=1)
    cmeta = cmeta.clone()
    cmeta[ctrl_missing] = -1.0
    x_data_in = torch.cat([x_data, control], dim=2).to(device)
    # append row 4 AFTER the control column is joined on, so the id spans assays + control alike
    x_meta_in = _with_cell_row(torch.cat([x_meta, cmeta], dim=2), gname, h5, model).to(device)
    dna = torch.tensor(np.array(g["dna"][np.sort(wi)]), dtype=torch.float32).to(device)   # [B,G,4]
    z = model.encode(x_data_in, dna, x_meta_in)             # [B, L2, d]
    return z.mean(dim=1).float().cpu().numpy() if pool else z


def _h5_dsf_levels(h) -> tuple:
    """DSF levels actually present in this h5, ascending.

    NEVER hardcode (1,2,4,8): dsf_list is a panel choice. A panel baked with [1,2,4] has no
    `meta_dsf8`/`counts_dsf8`, and assuming level 8 raises
    `KeyError: object 'meta_dsf8' doesn't exist` deep inside evaluation, after training has already run.
    """
    return tuple(sorted(int(k) for k in json.loads(h.attrs["dsf_list"])))


@torch.no_grad()
def eval_M3(model, h5_path, device, *, chrom: Optional[Sequence[str]] = None, n_regions: int = 8,
            batch_size: int = 4, dsf_levels=None, seed: int = 0,
            include_deprecated: bool = False) -> Dict:
    # `include_deprecated` is accepted for a uniform evaluate() call; M3 emits no deprecated keys.
    rng = np.random.default_rng(seed)
    model.eval()
    with _open_h5(h5_path) as h:
        dsf_levels = _h5_dsf_levels(h) if dsf_levels is None else tuple(dsf_levels)
        order = json.loads(h["biosamples"].attrs["order"])
        t_bios = [b for b in order if b.startswith("T_")]
        wpool = _window_pool(h, chrom)
        if wpool.size == 0 or not t_bios:
            return dict(ratio=float("nan"), within=float("nan"), between=float("nan"), n_regions=0)
        within, region_z1 = [], []
        for _ in range(n_regions):
            bio = t_bios[rng.integers(len(t_bios))]
            wi = np.sort(rng.choice(wpool, size=min(batch_size, wpool.size), replace=False))
            gname = bio.replace("/", "_")
            zs = [_encode_region_at_dsf(model, h, gname, wi, k, device) for k in dsf_levels]
            z1 = zs[0]
            region_z1.append(z1)
            for zk in zs[1:]:
                within.append(_cos_dist(zk, z1))              # same region, different input DSF
        within = np.concatenate(within) if within else np.zeros(0)
        Z1 = np.concatenate(region_z1) if region_z1 else np.zeros((0, 1))
        # region id per row of Z1, so the `between` pool can exclude SAME-REGION pairs -- without this
        # the "different biology" baseline is contaminated by pairs drawn from one region draw, which
        # deflates `between` and inflates the invariance ratio.
        region_ids = np.concatenate([np.full(len(z), i) for i, z in enumerate(region_z1)]) \
            if region_z1 else np.zeros(0, int)
        N = len(Z1)
        if N < 2 or within.size == 0:
            return dict(ratio=float("nan"), within=float("nan"), between=float("nan"), n_regions=n_regions)
        idx = rng.integers(0, N, size=(min(4000, N * 4), 2))
        idx = idx[idx[:, 0] != idx[:, 1]]
        idx = idx[region_ids[idx[:, 0]] != region_ids[idx[:, 1]]]
        if idx.shape[0] == 0:
            return dict(ratio=float("nan"), within=float(within.mean()), between=float("nan"),
                        n_regions=n_regions, encoder_eff_rank_pooled=effective_rank(Z1),
                        n_between_pairs=0)
        between = _cos_dist(Z1[idx[:, 0]], Z1[idx[:, 1]])
        wmean, bmean = float(within.mean()), float(between.mean())
        eff = effective_rank(Z1)
    return dict(within=wmean, between=bmean, ratio=wmean / (bmean + 1e-8),
                encoder_eff_rank_pooled=eff,
                invariance_ok=bool((wmean / (bmean + 1e-8)) <= 0.3 and eff > 1.0),
                n_regions=n_regions, n_between_pairs=int(idx.shape[0]))


# ---------------------------------------------------------------------------
# S14 — depth counterfactual scored against counts_dsf{k} (a REAL counterfactual ground truth)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _dsf_counterfactual(model, h5_path, device, assays: Sequence[str], *,
                        chrom: Optional[Sequence[str]] = None, n_windows: int = 4,
                        n_region_draws: int = 6, dsf_levels=None, fg_frac: float = 0.02,
                        seed: int = 0, reference=None) -> Dict:
    """Score each TOLD depth against its OWN ground truth `counts_dsf{k}`, not the fixed dsf1 one.

    The plain depth sweep compares every told depth against the dsf1 target, so telling a lower depth is
    wrong BY CONSTRUCTION and `dir_mean_delta > 0` is guaranteed for any model whose mu decreases with
    told depth -- which the offset does arithmetically. Measured discriminative power between arms:
    `frac_min_at_true` 0.7588 vs 0.7597.

    Here `counts_dsf{k}` + `meta_dsf{k}` supply a real counterfactual: the encoder input is held at the
    honest dsf1 T_ view, and only the PROMPT and the GROUND TRUTH move together to depth level k. The
    bar is `CRPS(told=k, GT=k) < CRPS(told=1, GT=k)` -- unsatisfiable by the offset unless the offset is
    right, and identical for both arms.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    A = len(assays)
    per_target = []
    with _open_h5(h5_path) as h:
        dsf_levels = _h5_dsf_levels(h) if dsf_levels is None else tuple(dsf_levels)
        order = json.loads(h["biosamples"].attrs["order"])
        wpool = _window_pool(h, chrom)
        t_bios = [b for b in order if b.startswith("T_")]
        for tb in t_bios:
            base = tb[2:]
            t_meta = np.array(h["biosamples"][tb]["meta_dsf1"])            # [4, F]
            for ib in [b for b in order if b[2:] == base and not b.startswith("T_")]:
                i_meta = np.array(h["biosamples"][ib]["meta_dsf1"])
                # held-out target = absent from the T_ input, present in the V_/B_ view
                for a in [c for c in range(A)
                          if float(t_meta[0, c]) == -1.0 and float(i_meta[0, c]) != -1.0]:
                    curves = {}
                    fired_any = False
                    for _ in range(n_region_draws):
                        wi = np.sort(rng.choice(wpool, size=min(n_windows, wpool.size), replace=False))
                        z = _encode_region_at_dsf(model, h, tb, wi, 1, device, pool=False)
                        B = z.shape[0]
                        # S14 builds its own decoder calls rather than going through `_decode`, so the
                        # reference offset has to be rebuilt here or the residual arm would be scored
                        # with half its mean model missing — and it would look catastrophically bad
                        # for a reason that has nothing to do with told depth.
                        lref = None
                        if reference is not None:
                            own = (np.asarray(h["biosamples"][tb]["counts_dsf1"][wi], dtype=np.float64)
                                   if reference.contributes(tb) else None)
                            lref = torch.as_tensor(reference.log_ref(wi, tb, own), device=z.device)
                        for k in dsf_levels:
                            gt = np.array(h["biosamples"][ib][f"counts_dsf{k}"][wi, :, a])   # [B, L]
                            fg, fired = _foreground_mask(gt.reshape(-1), fg_frac, seed=seed)
                            fired_any = fired_any or fired
                            for told in dsf_levels:                       # PROMPT at level `told`
                                tm = np.array(h["biosamples"][ib][f"meta_dsf{told}"])
                                ym = torch.full((B, 4, A), -1.0, device=z.device)
                                ym[:, :, a] = torch.tensor(tm[:, a], dtype=torch.float32,
                                                           device=z.device)
                                ym = _with_cell_row(ym, ib, h, model)
                                out = decode_latent(model, z, ym, lref)
                                mu = out["mu"][:, :, a].reshape(-1).double().cpu().numpy()
                                n = out["n"][:, :, a].reshape(-1).double().cpu().numpy()
                                c = float(np.mean(nb_crps(n[fg], _p_from_true_mu(n[fg], mu[fg]),
                                                          gt.reshape(-1)[fg])))
                                curves.setdefault((k, told), []).append(c)
                    if not curves:
                        continue
                    M = {kk: float(np.mean(v)) for kk, v in curves.items()}
                    wins = [(k, M[(k, k)] < M[(k, 1)]) for k in dsf_levels if k != 1]
                    per_target.append(dict(
                        target=(tb, ib, assays[a]),
                        crps_matrix={f"gt{k}_told{t}": M[(k, t)] for k in dsf_levels for t in dsf_levels},
                        min_at_true={f"gt{k}": bool(min(dsf_levels, key=lambda t: M[(k, t)]) == k)
                                     for k in dsf_levels},
                        beats_told1={f"gt{k}": bool(w) for k, w in wins},
                        purity_fallback_fired=bool(fired_any),
                        n_fg=int(fg.sum())))
    if not per_target:
        return dict(n_targets=0, frac_min_at_true=float("nan"), frac_beats_told1=float("nan"))
    mat = [v for t in per_target for v in t["min_at_true"].values()]
    beat = [v for t in per_target for v in t["beats_told1"].values()]
    return dict(per_target=per_target, n_targets=len(per_target),
                frac_min_at_true=float(np.mean(mat)), frac_beats_told1=float(np.mean(beat)),
                dsf_counterfactual_ok=bool(np.mean(beat) > 0.5))


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------

def _split_of(imp_biosample: str) -> str:
    """`V_K562` -> `V_`. The EIC split a target belongs to; the h74 gate is `V_` only."""
    return imp_biosample[:2] if imp_biosample[:2] in ("V_", "B_") else "?"


@torch.no_grad()
def reference_only_baseline(units, assays: Sequence[str], *, fg_frac: float = 0.02, seed: int = 0,
                            depth_center: float = 0.0) -> Dict:
    """Score the AVERAGE REFERENCE ALONE as a forecast — h27's actual bar, never previously measured.

    The `marg_crps` baseline the kit already reports is one CONSTANT distribution per assay. h27's
    claim is about the per-position average ACROSS CELL TYPES, which is a far stronger predictor, and
    no candi run has ever been scored against it. Without this number a model can look skilful
    while doing nothing the average does not already do — precisely the failure q23 exists to test.

    The forecast is `mu = R * 2^(d - depth_center)` with the same depth prompt the model receives, and
    a CRPS-OPTIMAL dispersion fitted per target. Granting it an oracle dispersion makes it a STRONG
    bar deliberately: it should not lose on a nuisance parameter neither arm is being asked about.

    SCORED OVER ALL POSITIONS IN `imp_map`, with NO foreground mask, because that is what
    `_suite_light` does for `imp_per_target["crps"]` — the quantity V1 compares and the only thing
    this number is useful next to. Scoring the reference on the top-2% foreground instead makes it
    look ~9x worse than the trained arms purely because the foreground carries much larger counts;
    that comparison is meaningless and was made once before this note existed. `crps_fg` is reported
    alongside as a LABELLED second measurement, since the foreground is where cell-specific peaks
    live and the average is weakest there by construction.

    Requires units built with a reference (`u.log_ref` present); returns `available=False` otherwise.
    """
    per_target: List[Dict] = []
    if not any(u.log_ref is not None for u in units):
        return dict(available=False)
    for u in units:
        if u.imp_map is None or u.log_ref is None or u.y_meta_imp is None:
            continue
        for a in u.imp_target_assays():
            m = u.imp_map[:, :, a]
            if not bool(m.any()):
                continue
            # invert the offset: log_ref = log2(R + c) -> R, then apply the target's own exposure
            d = u.y_meta_imp[:, DEPTH_ROW, a]
            if not bool(torch.isfinite(d).all()) or bool((d < 0).any()):
                continue
            log_mu = u.log_ref[:, :, a] + (d - depth_center).unsqueeze(1)
            mu = torch.pow(2.0, log_mu.clamp(-15.0, 30.0))[m].double().cpu().numpy()
            tgt = u.y_data_imp[:, :, a][m].double().cpu().numpy()
            if mu.size < 2:
                continue

            def _best(mf, tf):
                # oracle DISPERSION only — the location is the reference's own, ungranted
                return min((float(np.mean(nb_crps(np.full_like(mf, nv), _p_from_true_mu(
                    np.full_like(mf, nv), mf), tf))), nv)
                    for nv in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0))

            crps, n_star = _best(mu, tgt)
            fg, fired = _foreground_mask(tgt, fg_frac, seed=seed)
            crps_fg = _best(mu[fg], tgt[fg])[0] if int(fg.sum()) >= 2 else float("nan")
            per_target.append(dict(target=(u.biosample, u.imp_biosample, assays[a]),
                                   crps=crps, n_star=n_star, crps_fg=crps_fg,
                                   n_points=int(mu.size), n_fg=int(mu.size),
                                   purity_fallback_fired=False))
    out = dict(available=True, n_targets=len(per_target), per_target=per_target)
    for split in ("V", "B"):
        rs = [r for r in per_target if r["target"][1][:1] == split]
        for key in ("crps", "crps_fg"):
            usable = [r for r in rs if np.isfinite(r.get(key, np.nan))]
            c = _cluster_bootstrap_ci(usable, value_key=key, n_boot=1, seed=seed) if usable else None
            out[f"{split}_{key}"] = c["mean"] if c else float("nan")
            if key == "crps":
                out[f"{split}_n_targets"] = c["n_clusters"] if c else 0
    return out


@torch.no_grad()
def quick_eval(model, ds, device, *, batches_per_pair: int = 2, fg_frac: float = 0.02,
               seed: int = 0, meta_probe=None) -> dict:
    """Mid-training imputation scorer: per-target foreground CRPS, split `V_`/`B_`. Cheap.

    Takes a ready eval dataset (t28). The caller opens it — `train.py` through `make_dataset`, an
    h5 caller through `open_h5_eval_dataset` — which is what lets a store-backed run be scored at
    all. Opening it ONCE and re-iterating it every check is also what keeps the window set fixed
    across epochs, so epoch-to-epoch comparison stays paired.

    `seed` no longer reaches the loader; it seeds the foreground draw only. The loader's own seed is
    a property of the dataset the caller built.

    Exists because h73 ran a single evaluation AFTER training and could therefore not distinguish
    over- from under-fitting. This runs every few epochs and drives best-checkpoint selection.

    Deliberately NOT `evaluate`: no M2, no M3, no S14, no bootstrap, no oracle-scale, no marginal
    baseline. Point estimates are all checkpoint selection needs, and the full instrument costs about
    an hour. Coverage is trimmed by `batches_per_pair`, which keeps EVERY target and shrinks the
    windows per target, so the number is comparable across epochs and across arms.

    SELECTION IS ON ALL POSITIONS, not the foreground. `imp_per_target["crps"]` — the quantity V1
    compares — is computed by `_suite_light` over the whole target, so selecting on a foreground CRPS
    would optimise a different objective than the one being scored and could hand the final eval a
    checkpoint that is good at peaks and worse overall. The foreground number is reported alongside
    as `*_imp_crps_fg` because it is where cell-specific biology lives, but it does not drive
    selection. The collapse is `_cluster_bootstrap_ci`'s, so the selection metric is the V1 metric on
    fewer windows rather than a second, differently-defined number.

    `meta_probe` (h64) threads the arm's row-3 transform through this eval as well, so best-checkpoint
    selection is made against the SAME objective the arm trains on. None (the `off` arm) is a strict
    no-op. The FINAL `evaluate` deliberately does not apply it — see the note at its call site in
    `train.py` and the `final_eval_meta_probe_applied` key in the run JSON.

    NEVER AUTOCAST, whatever `--precision` the run trains at — see `evaluate` for why, and note the
    sharper reason here: this function drives BEST-CHECKPOINT SELECTION. A CRPS measured at one
    precision at epoch 6 and another at epoch 12 would select on the difference between the two.
    """
    with no_autocast(device):
        return _quick_eval_fp32(model, ds, device, batches_per_pair=batches_per_pair,
                                fg_frac=fg_frac, seed=seed, meta_probe=meta_probe)


def _quick_eval_fp32(model, ds, device, *, batches_per_pair: int = 2, fg_frac: float = 0.02,
                     seed: int = 0, meta_probe=None) -> dict:
    units, assays = build_eval_units(model, ds, device, batches_per_pair=batches_per_pair,
                                     meta_probe=meta_probe)
    recs: List[Dict] = []
    for u in units:
        if u.imp_map is None:
            continue
        out = _decode(model, u, u.y_meta_true)
        for a in u.imp_target_assays():
            g = _gather(out, u.imp_map, a, u.y_data_imp)
            if g is None:
                continue
            fg, _fired = _foreground_mask(g["target"], fg_frac, seed=seed)
            recs.append(dict(
                target=(u.biosample, u.imp_biosample, assays[a]),
                split=_split_of(u.imp_biosample),
                # weight by ALL positions, matching how `compare_arms` weights `n_points`
                n_fg=int(g["target"].size), purity_fallback_fired=False,
                crps=float(np.mean(nb_crps(g["n"], g["p"], g["target"]))),
                crps_fg=(float(np.mean(nb_crps(g["n"][fg], g["p"][fg], g["target"][fg])))
                         if int(fg.sum()) >= 2 else float("nan"))))

    def _mean(rs, key):
        # n_boot=1 because only `mean` is read; the interval is meaningless at that size and is not
        # returned. Reusing the function guarantees the collapse matches the scoring path.
        rs = [r for r in rs if np.isfinite(r.get(key, np.nan))]
        if not rs:
            return dict(mean=float("nan"), n_clusters=0)
        c = _cluster_bootstrap_ci(rs, value_key=key, n_boot=1, seed=seed)
        return dict(mean=c["mean"], n_clusters=c["n_clusters"])

    out: Dict = dict(n_units=len(units), n_records=len(recs))
    for split in ("V", "B"):
        rs = [r for r in recs if r["split"] == f"{split}_"]
        a, f = _mean(rs, "crps"), _mean(rs, "crps_fg")
        out[f"{split}_imp_crps"] = a["mean"]          # SELECTION metric — matches V1
        out[f"{split}_n_targets"] = a["n_clusters"]
        out[f"{split}_imp_crps_fg"] = f["mean"]       # reported, never selected on
    return out


@torch.no_grad()
def evaluate(model, h5_path, device, *, regime: str = "type1", batch_size: int = 4,
             max_batches: Optional[int] = None, fg_frac: float = 0.02, n_boot: int = 1000,
             seed: int = 0, eval_budget: int = 200_000, m3_regions: int = 8,
             include_deprecated: bool = False, reference=None) -> dict:
    """EVALUATION IS NEVER AUTOCAST, whatever `--precision` the run was trained at.

    Every recorded number in this repo — the Gate B acceptance bands, the CRPS/Spearman/ECE anchors,
    the M2 slopes — was measured in fp32. A metric measured at a different precision than the one it
    is compared against is an undeclared difference between arms, and it would land in exactly the
    third decimal place those bands are set at. Training precision is a throughput choice; scoring
    precision is part of the measurement, and this repo has one.

    The fence is here rather than at each of the dozen forwards below because the guarantee wanted is
    "nothing under this call", not "not this line". `candi.metrics` needs no fence at all: it is
    numpy, and autocast is a torch dispatcher feature that never reaches it.
    """
    with no_autocast(device):
        return _evaluate_fp32(model, h5_path, device, regime=regime, batch_size=batch_size,
                              max_batches=max_batches, fg_frac=fg_frac, n_boot=n_boot, seed=seed,
                              eval_budget=eval_budget, m3_regions=m3_regions,
                              include_deprecated=include_deprecated, reference=reference)


def _evaluate_fp32(model, h5_path, device, *, regime: str = "type1", batch_size: int = 4,
                   max_batches: Optional[int] = None, fg_frac: float = 0.02, n_boot: int = 1000,
                   seed: int = 0, eval_budget: int = 200_000, m3_regions: int = 8,
                   include_deprecated: bool = False, reference=None) -> dict:
    # `evaluate` is the h5-only half of this module, so it opens its own h5 dataset and hands it to
    # the shared builder. Store-backed scoring does not come through here.
    eval_ds = open_h5_eval_dataset(model, h5_path, regime=regime, batch_size=batch_size,
                                   seed=seed, reference=reference)
    units, assays = build_eval_units(model, eval_ds, device, max_batches=max_batches)
    s14 = _dsf_counterfactual(model, h5_path, device, assays, fg_frac=fg_frac, seed=seed,
                              reference=reference)
    # Both calibrations, inline, because neither is obvious from the number:
    print(f"[S14] frac_min_at_true={s14.get('frac_min_at_true')} "
          f"frac_beats_told1={s14.get('frac_beats_told1')} n_targets={s14.get('n_targets')}\n"
          f"[S14] calibration 1: 0.25 is NOT chance — it is the deterministic value of "
          f"argmin-always-at-told=1.\n"
          f"[S14] calibration 2: a perfect model caps at frac_min_at_true ~= 0.73, because the "
          f"foreground is the top {fg_frac:.0%} of the level-k realization being scored.", flush=True)
    ref_bar = reference_only_baseline(
        units, assays, fg_frac=fg_frac, seed=seed,
        depth_center=float(getattr(getattr(model, "decoder", None), "depth_center", 0.0)))
    if ref_bar.get("available"):
        print(f"[h27bar] reference-alone forecast: V_ crps={ref_bar['V_crps']:.4f} "
              f"(n={ref_bar['V_n_targets']})  B_ crps={ref_bar['B_crps']:.4f} "
              f"(n={ref_bar['B_n_targets']}). This is the average epigenome with an oracle "
              "dispersion — the bar a trained model has to clear to have learned any deviation.",
              flush=True)
    return dict(
        n_units=len(units),
        assays=list(assays),
        reference_only_baseline=ref_bar,
        M1=eval_M1(model, units, device, assays, budget=eval_budget, seed=seed,
                   include_deprecated=include_deprecated),
        M2=eval_M2(model, units, device, assays, fg_frac=fg_frac, n_boot=n_boot, seed=seed,
                   include_deprecated=include_deprecated),
        M3=eval_M3(model, h5_path, device, n_regions=m3_regions, seed=seed,
                   include_deprecated=include_deprecated),
        S14=s14,
    )


def main() -> None:
    """Score an already-trained checkpoint. Scale (num_assays, context_bins, assay order, eval chroms)
    comes from the h5; the architecture flags must match the ones the checkpoint was trained with, or
    the strict load below fails loudly."""
    import argparse

    from candi.dataset import h5_depth_center
    from candi.model import build_model, build_model_from_arch

    ap = argparse.ArgumentParser(description="Evaluate a trained candi checkpoint (M1/M2/M3/S14)")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="results JSON to write")
    ap.add_argument("--offset", "--arm", dest="offset", default="on",
                    choices=["on", "off", "offset_on", "offset_off"])
    ap.add_argument("--regime", default="type1", choices=["type1", "type2_loci"])
    ap.add_argument("--depth-center", type=float, default=None)
    ap.add_argument("--d-model", type=int, default=0)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--n-transformer-layers", type=int, default=2)
    ap.add_argument("--decoder-lane", type=int, default=8)
    ap.add_argument("--deconv-norm", default="lane", choices=["lane", "group"])
    ap.add_argument("--arch-from", default=None,
                    help="a run's own JSON. Reads config.arch and rebuilds the EXACT model that "
                         "wrote the checkpoint, so none of the architecture flags above need to be "
                         "retyped and none of them can be retyped WRONG. Prefer this: every "
                         "geometry, norm, FiLM and head flag changes the state_dict, and a "
                         "mismatched one shows up as a strict-load failure at best and as a "
                         "quietly different model at worst. Overrides the flags it covers.")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = all")
    ap.add_argument("--fg-frac", type=float, default=0.02)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--eval-budget", type=int, default=200_000)
    ap.add_argument("--m3-regions", type=int, default=8)
    ap.add_argument("--include-deprecated", action="store_true")
    ap.add_argument("--reference", default="off", choices=["off", "on"],
                    help="MUST match how the checkpoint was trained. The final eval is the last hour "
                         "of a ~7h run and the checkpoint is written BEFORE it, so a crash there is "
                         "recoverable — but only if this flag exists, or a residual-arm checkpoint "
                         "could only be re-scored without half its mean model.")
    ap.add_argument("--reference-path", default=None)
    ap.add_argument("--reference-pseudocount", type=float, default=None)
    ap.add_argument("--meta-embed-layernorm", default="on", choices=["on", "off"],
                    help="MUST match how the checkpoint was trained, for the same reason --reference "
                         "must: an h75 ln_off checkpoint has no fusion LayerNorm parameters, so the "
                         "strict load below fails and the arm cannot be re-scored without it.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ds = CandiKitH5Dataset(a.h5, a.regime, train=False, batch_size=a.batch_size, h5_cache_ram=False)
    depth_center = a.depth_center if a.depth_center is not None else h5_depth_center(a.h5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.arch_from:
        arch = (json.loads(Path(a.arch_from).read_text()).get("config") or {}).get("arch")
        if not arch:
            raise SystemExit(f"{a.arch_from} has no config.arch block — it predates --arch-from. "
                             "Pass the architecture flags by hand, matching the run's config.")
        # --depth-center still wins if given explicitly: it is a data fact, not an architecture one,
        # and re-scoring against a rebuilt reference is a legitimate reason to override it.
        if a.depth_center is not None:
            arch["depth_center"] = float(a.depth_center)
        model = build_model_from_arch(arch).to(device)
        print(f"[eval] architecture read from {a.arch_from}: "
              f"{sum(p.numel() for p in model.parameters()):,} params", flush=True)
    else:
        model = build_model(embed_dim=a.embed_dim, dropout=a.dropout,
                            n_transformer_layers=a.n_transformer_layers,
                            decoder_lane=a.decoder_lane, deconv_norm=a.deconv_norm,
                            depth_center=depth_center,
                            use_offset=a.offset in ("on", "offset_on"),
                            num_assays=ds.num_assays, context_length=ds.context_bins,
                            resolution=int(ds.resolution),
                            d_model=a.d_model, nhead=a.nhead,
                            meta_embed_layernorm=(a.meta_embed_layernorm == "on")).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device), strict=True)
    model.eval()

    ref = None
    if a.reference == "on":
        from candi.reference import REFERENCE_PSEUDOCOUNT, ReferenceTable, reference_path_for
        ref = ReferenceTable(a.reference_path or reference_path_for(a.h5), src_h5=a.h5,
                             pseudocount=(REFERENCE_PSEUDOCOUNT if a.reference_pseudocount is None
                                          else a.reference_pseudocount))
        if abs(ref.depth_center - depth_center) > 1e-6:
            raise ValueError(f"depth_center mismatch: reference {ref.depth_center:.6f} vs eval "
                             f"{depth_center:.6f}; the two offsets would compose on different scales")
        print(f"[eval] reference=on: {ref.path}, {len(ref.contributors)} contributors", flush=True)

    res = evaluate(model, a.h5, device, regime=a.regime, batch_size=a.batch_size,
                   max_batches=(a.max_batches or None), fg_frac=a.fg_frac, n_boot=a.n_boot,
                   seed=a.seed, eval_budget=a.eval_budget, m3_regions=a.m3_regions,
                   include_deprecated=a.include_deprecated, reference=ref)
    from candi.train import _jsonable
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(_jsonable(res), f, indent=2)
    print(f"[eval] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
