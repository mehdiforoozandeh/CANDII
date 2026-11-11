"""E6 -- the onion scorer: a miniCANDI run or a training-free baseline -> `scores.json` (pinned interface D).

    python tools/onion_eval/score.py (--run <run dir> | --baseline <name>) --evalset <json> --out <dir>
        [--split val|blind] [--allow-blind] [--repeat-blind-reason "<text>"] [--no-halves]
        [--dnase-route hotspot2_density|macs2_no_control] [--device cpu] [--n-boot 200]
        [--guardrail-windows 256] [--force]
    -> <out>/scores.json, <out>/suffstats.npz, <out>/SCORED (SCORED last; an existing SCORED skips
       the job unless --force). The scorer never reads or writes DONE: training's DONE in a run dir
       means trained, not scored (finding 3).

A baseline is scored into `eval/<tables>/baselines/<name>/seed0/` so `select_rule.py table` reads it
like a run. Baselines: `marginal_nb`, `position_average`, `depth_only_scaling` (E4, the three
training-free baselines of D14) and `pseudoreplicate_floor` (the half-against-half floor of headline
4; a reference, not a D14 baseline). By definition depth-only scores a correction gap closed of
exactly 0, the floor exactly 1, and marginal NB an uncertainty term of exactly 0.

What is scored (validation split by default):
  imputation targets  per regime (`leave_one_assay_out` is the headline; `accessibility_only`,
                      D11b.2, is the second table under the key `accessibility_only`). The ChIP-seq
                      control is never an imputation target; when the model predicts the control
                      (`predict_control`), the evalset's `control_targets` are scored in a separate
                      table under `predicted_control` (finding 16)
  correction pairs    source pr1 -> target pr2 (PI ruling 6); model, depth-only scaling and the
                      floor (target pr1 -> target pr2, one fitted NB size) all predict the same pr2
  replicate pairs     the evalset's replicate-ladder pairs of the scored split, same layout
  copy-leak           guardrail parts 1 and 2 (`guardrail.copy_leak`) on the validation
                      reconstruction windows, over 3 mask seeds. `copy_leak_pass` is true only when
                      every mask seed passes (finding 2); `copy_leak_spread_ok` is true only when
                      the part-2 relative gain moved across mask seeds by at most
                      SPREAD_MAX_FRACTION x the gain threshold (addendum); select_rule escalates
                      an arm whose spread is not ok

Blind discipline (gate G4; finding 9 and 14). `--split blind` needs `--allow-blind`, an eval set
built with `--include-blind` (pinned name `evalset_blind.json`) and `--out` ending in `/blind`
(`<runs>/<arm>/seed<k>/blind`, `eval/<tables>/baselines/<name>/seed0/blind`). Blind rows are
accepted only when `split == blind`. Every blind pass is appended to the append-only ledger
`BLIND_PASSES.tsv` beside the eval set (time, run, evalset sha256, tables sha256, git sha, out,
status, repeat reason); a second pass for the same (run, eval set) is refused unless
`--repeat-blind-reason "<text>"` is given, and the reason is recorded.

Tables check (finding 9). The eval set's `tables_sha256` must equal `minicandi.dataset.tables_sha256`
of the tables on disk, and on validation the run checkpoint's `tables_sha256` too (the blind pass
reads eic_v2, a superset of the training tables, so there the checkpoint's value is recorded as
`run_tables_sha256`, not required equal).

Pseudoreplicate halves (finding 11). A target without halves (`has_pseudoreps` not 1, or a
`store:` path) gets noise floor N = Undefined with the reason, is kept out of headline 1 and counted
(`n_targets_without_halves`). Headline 1 is Undefined unless E_sp, V_sp and N are all defined on
every included target (finding 6). `--no-halves` (the pilot on `store:` tables): no half is ever
read; `specific_variance_recovered_above_noise` is null with the reason, and headline 1 without a
noise floor is written under `specific_variance_recovered_NO_NOISE_FLOOR_pilot_only`;
`noise_floor = "none_pilot"`, which select_rule refuses.

Every metric is computed by the merged library (`metrics.py`, `error_budget.py`), once per target
track over all of its bins (addendum): the scorer streams the predictor region by region and tile by
tile, assembles each target's vectors over the evalset regions (whole 2^14-bin blocks, so the Haar
transform never mixes two regions), computes that target's scalars, and keeps the vectors only
while its assay group is open (headline 1 and the error budget split shared from specific across the
targets of one assay). `suffstats.npz` keeps the sufficient statistics: the Gram matrices per Haar
scale (shared and specific parts of V, P, C, E), the PIT curves (F_bar on the 21-point grid, with bin
counts, per target and stratum), and the CRPS sums per target and pair.

Space of headline 1 and 2 for selection: always `counts_log1p` (ruling 13; there is no option).
`signal_arcsinh` is computed through `signal_from_counts.signal_from_nb` (X2), imported lazily, with
the dataset, the region list and the region length; a ChIP-seq target without an observed control
uses the model's predicted control (a second forward pass with the control channel asked for at
the target's own C'). `lambda_source` per target is written under `per_target`. `--dnase-route`
selects the DNase-seq route (default hotspot2_density, D9b). Without the module the signal space is
NaN with the reason.

Per-metric aggregation. Each metric is {"macro", "per_assay", "ci", "non_scorable_per_assay",
"missing_scorable", "reason"}: per assay over that assay's targets (headline 1 and 4 as ratios of
means, the others as means), macro = mean over the scorable assays (evalset `scorable`), undefined
when any scorable assay is undefined, and in the leave-one-assay-out headline and the correction
metrics also when a scorable assay has no target (finding 10). `ci` is a 95% cluster bootstrap over
cells (targets resampled with their cell; the per-target components are held fixed, so headline 1's
shared estimate is not refitted per draw).

Selection denominators (finding 4). `selection.denominators` holds the marginal-NB PIT deviation
(uncertainty) and, per correction-scorable assay, (depth-only CRPS - floor CRPS) / depth-only CRPS,
with its minimum; select_rule escalates a term whose denominator is under its tolerance, or a term
outside its sanity band.

Baseline fitting (agent decisions, logged): marginal NB per assay and the NB size of depth-only
scaling and of the floor are fitted on the evalset's `fit_regions` (never a scored bin); the
position average's leave-one-cell-out table and its NB size are fitted per (assay, region) on the
training cells' base views of that region (the table must cover the scored bins and the genome-wide
table does not fit in memory), so its NB size is one per assay per region. The position average is
also the training average `a` of r_spec.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_eval import metrics as Mx  # noqa: E402

SPACE = "counts_log1p"                               # ruling 13: the selection space, fixed
REFS = ("marginal_nb", "position_average", "depth_only_scaling", "pseudoreplicate_floor")
TRAINING_FREE = ("marginal_nb", "position_average", "depth_only_scaling")
WEIGHTS = {"imputation": 0.5, "correction": 0.25, "uncertainty": 0.25}
REGIMES = ("leave_one_assay_out", "accessibility_only")
REPLICATE_RUNGS = ("pseudoreplicate", "sequencing", "technical", "isogenic", "anisogenic",
                   "other_experiment", "other_cell_type")
POOLED = "__pooled__"
BUDGET_KEYS = ("level", "shared", "specific", "explained", "unexplained", "noise", "added")
CONTROL = "chipseq-control"
NOISE_FLOOR_HALVES = "pseudoreplicate_halves"
NOISE_FLOOR_NONE = "none_pilot"
PILOT_H1_KEY = "specific_variance_recovered_NO_NOISE_FLOOR_pilot_only"
#: copy-leak part 2: the relative gain's spread across mask seeds must be at most this fraction of
#: the gate threshold ("well under", addendum), else `copy_leak_spread_ok` is false
SPREAD_MAX_FRACTION = 0.5
BLIND_LEDGER = "BLIND_PASSES.tsv"
LEDGER_COLUMNS = ("time", "run", "evalset_sha256", "tables_sha256", "git_sha", "out", "status",
                  "repeat_reason")
SCORED = "SCORED"


class ScoreRefused(Exception):
    """The scorer may not run on these inputs (blind discipline, tables mismatch, layout)."""


# ------------------------------------------------------------------------------------------ helpers

def _undef(x) -> bool:
    return x is None or not np.isfinite(float(x))


def _j(x):
    """JSON value: finite float, or None for an undefined one."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _why(x) -> str:
    return getattr(x, "reason", "") if x is not None else "missing"


def composite(imp, cor, unc):
    """D13: 0.5 imputation + 0.25 correction + 0.25 uncertainty; Undefined if a term is."""
    terms = {"imputation": imp, "correction": cor, "uncertainty": unc}
    bad = [k for k, v in terms.items() if _undef(v)]
    if bad:
        return Mx.Undefined("undefined goal term: " + ", ".join(bad))
    return float(sum(WEIGHTS[k] * float(v) for k, v in terms.items()))


def _macro(per: dict):
    if not per:
        return Mx.Undefined("no scorable assay with a target")
    bad = [a for a, v in per.items() if _undef(v)]
    if bad:
        return Mx.Undefined("undefined for " + ", ".join(f"{a} ({_why(per[a])})" for a in bad))
    return float(np.mean([float(v) for v in per.values()]))


def _boot(records, agg, n_boot, seed):
    cells = sorted({r["cell"] for r in records})
    if len(cells) < 2 or n_boot <= 0:
        return [None, None]
    by = defaultdict(list)
    for r in records:
        by[r["cell"]].append(r)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(cells), len(cells), replace=True)
        v = agg([r for i in pick for r in by[cells[i]]])
        if not _undef(v):
            vals.append(float(v))
    if len(vals) < max(10, n_boot // 2):
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def metric(records, fn, scorable, n_boot=200, seed=0, undefined_reason=None,
           require_all=False) -> dict:
    """{"macro", "per_assay", "ci", "non_scorable_per_assay", "missing_scorable", "n", "reason"}
    from per-target records. `require_all`: the macro is Undefined when a scorable assay has no
    record (finding 10); `missing_scorable` lists such assays in every case."""
    if undefined_reason:
        return {"macro": None, "per_assay": {}, "ci": [None, None], "non_scorable_per_assay": {},
                "missing_scorable": [], "n": 0, "reason": undefined_reason}
    scorable = set(scorable)

    def per_assay(recs):
        by = defaultdict(list)
        for r in recs:
            by[r["assay"]].append(r)
        return {a: fn(rs) for a, rs in sorted(by.items())}

    def agg(recs):
        return _macro({a: v for a, v in per_assay(recs).items() if a in scorable})

    per = per_assay(records)
    sc = {a: v for a, v in per.items() if a in scorable}
    missing = sorted(scorable - set(per))
    macro = _macro(sc)
    if require_all and missing and not _undef(macro):
        macro = Mx.Undefined("scorable assay with no target in this split and regime: "
                             + ", ".join(missing))
    return {"macro": _j(macro), "per_assay": {a: _j(v) for a, v in sc.items()},
            "ci": _boot(records, agg, n_boot, seed) if not _undef(macro) else [None, None],
            "non_scorable_per_assay": {a: _j(v) for a, v in per.items() if a not in scorable},
            "per_assay_reason": {a: _why(v) for a, v in per.items() if _undef(v)},
            "missing_scorable": missing,
            "n": len(records), "reason": _why(macro) if _undef(macro) else ""}


def _mean_of(key):
    def fn(rs):
        v = [r[key] for r in rs if not _undef(r[key])]
        if len(v) < len(rs):
            bad = [r for r in rs if _undef(r[key])]
            return Mx.Undefined(f"{len(bad)} of {len(rs)} targets undefined: {_why(bad[0][key])}")
        return float(np.mean(v))
    return fn


def recovered_fn(prefix: str = "", no_floor: bool = False):
    """Headline 1 of one assay's records (findings 6 and 11). Targets without pseudoreplicate halves
    are left out; every included target needs E_sp, V_sp and N defined, else Undefined with the
    reasons. `no_floor`: the pilot form, N = 0 for every target."""
    kE, kV, kN = f"{prefix}E_sp", f"{prefix}V_sp", f"{prefix}N"

    def fn(rs):
        inc = rs if no_floor else [r for r in rs if r.get("has_halves", True)]
        if not inc:
            return Mx.Undefined(f"no target with pseudoreplicate halves ({len(rs)} without)")
        keys = (kE, kV) if no_floor else (kE, kV, kN)
        bad = sorted({f"{k}: {_why(r[k])}" for r in inc for k in keys if _undef(r[k])})
        if bad:
            return Mx.Undefined("; ".join(bad))
        N = [0.0] * len(inc) if no_floor else [r[kN] for r in inc]
        return Mx.specific_variance_recovered([r[kE] for r in inc], [r[kV] for r in inc], N)
    return fn


def pit_curve(n, mu, y):
    from candi.bench.distributional import p_from_mu
    from candi.metrics import PIT_GRID, calibration_pit_curve
    _, fbar = calibration_pit_curve(n, p_from_mu(n, mu), y, PIT_GRID)
    return np.asarray(fbar, np.float64)


def pooled_pit_deviation(curves):
    """Mean |F_bar(u) - u| of the bin-weighted mean of several F_bar curves [(fbar, count)].
    Exactly `metrics.pit_deviation(...)["pooled"]` on the concatenated bins (F_bar is a bin mean)."""
    from candi.metrics import PIT_GRID
    curves = [(f, c) for f, c in curves if c > 0]
    if not curves:
        return Mx.Undefined("no bins")
    w = np.array([c for _, c in curves], np.float64)
    f = np.stack([f for f, _ in curves]) * w[:, None]
    return float(np.mean(np.abs(f.sum(0) / w.sum() - np.asarray(PIT_GRID))))


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(_TOOLS), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - no git on the node
        return "unknown"


def has_halves(row: dict) -> bool:
    """The row's pseudoreplicate halves exist on disk (a `store:` path never has them)."""
    return (str(row.get("has_pseudoreps", "")).strip() in ("1", "true", "True")
            and str(row.get("counts_path", "")).startswith("h5:"))


def _signal_module():
    try:
        from onion_eval import signal_from_counts as S  # noqa: F401  (X2, lazy)
    except ImportError as e:
        return None, f"tools/onion_eval/signal_from_counts.py (X2) not importable: {e}"
    if not hasattr(S, "signal_from_nb"):
        return None, "signal_from_counts has no signal_from_nb(n, mu, row, *, ds, regions, rb)"
    return S, ""


# ------------------------------------------------------------------------------------------ predictors

class ModelPredictor:
    def __init__(self, model, device="cpu"):
        self.model, self.device = model.to(device).eval(), device

    def __call__(self, batch):
        b = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            o = self.model(b)
        return {k: o[k].detach().cpu().double().numpy() for k in ("n", "mu")}


def _np_out(o):
    return {k: (o[k].detach().cpu().double().numpy() if torch.is_tensor(o[k])
                else np.asarray(o[k], np.float64)) for k in ("n", "mu")}


# ------------------------------------------------------------------------------------------ scorer

class Scorer:
    def __init__(self, evalset, *, main, model=None, config=None, split="val", allow_blind=False,
                 n_boot=200, seed=0, guardrail_windows=256, guardrail_seeds=(0, 1, 2),
                 device="cpu", no_halves=False, dnase_route=None, run_tables_sha256=None,
                 log=print):
        from minicandi import contract as C
        from minicandi.dataset import OnionDataset, tables_sha256
        from onion_eval.evalset import tables_dir_of
        if split not in ("val", "blind"):
            raise ValueError("split must be val or blind")
        if split == "blind" and not allow_blind:
            raise PermissionError("scoring blind targets needs --allow-blind (gate G4)")
        if main not in REFS and model is None:
            raise ValueError(f"main {main!r}: give a model or one of {REFS}")
        self.C, self.log = C, log
        self.evalset_path = Path(evalset).resolve()
        self.ev = json.loads(self.evalset_path.read_text(encoding="utf-8"))
        if split == "blind" and not self.ev.get("include_blind"):
            raise ScoreRefused(f"{self.evalset_path}: not a blind eval set (built without "
                               f"--include-blind)")
        self.tables = tables_dir_of(self.ev, self.evalset_path)
        self.tables_sha256 = tables_sha256(self.tables)
        if not self.ev.get("tables_sha256"):
            raise ScoreRefused(f"{self.evalset_path}: no tables_sha256 (rebuild the eval set)")
        if self.ev["tables_sha256"] != self.tables_sha256:
            raise ScoreRefused(f"tables {self.tables} changed since the eval set was built "
                               f"(sha256 {self.tables_sha256[:12]} vs {self.ev['tables_sha256'][:12]})")
        self.run_tables_sha256 = run_tables_sha256
        if model is not None and split == "val" and run_tables_sha256 != self.tables_sha256:
            raise ScoreRefused(f"the run was trained on tables sha256 {str(run_tables_sha256)[:12]}, "
                               f"the eval set's tables are {self.tables_sha256[:12]}")
        # finding 14: blind rows only on a blind pass
        self.split, self.allow_blind, self.n_boot, self.seed = split, split == "blind", n_boot, seed
        self.no_halves = bool(no_halves)
        self.main, self.model_pred = main, (ModelPredictor(model, device) if model is not None else None)
        self.guardrail_windows, self.guardrail_seeds = guardrail_windows, tuple(guardrail_seeds)
        self.model = model
        self.rb = int(self.ev.get("region_bins", 2 ** 14))
        cfg = dict(config or {})
        L = int(cfg.get("context_bins") or min(1024, self.rb))
        if self.rb % L:
            raise ValueError(f"region_bins {self.rb} is not a multiple of context_bins {L}")
        cfg.update({"context_bins": L, "evalset": str(self.evalset_path), "region_bins": self.rb,
                    "allow_blind": self.allow_blind})
        cfg.setdefault("batch_size", 8)
        self.cfg, self.L, self.bs = cfg, L, int(cfg["batch_size"])
        self._OnionDataset = OnionDataset
        self.ds = OnionDataset(self.tables, split, cfg, seed=0)
        self.predict_control = bool(self.ds.config.get("predict_control")) and model is not None
        self.regions = [(c, int(s)) for c, s in self.ev["regions"]]
        fr = [(c, int(s)) for c, s in self.ev.get("fit_regions", [])]
        self.fit_on = "fit_regions" if fr else "eval_regions (no fit_regions in the evalset)"
        self.fit_regions = fr or self.regions
        for c, s in self.regions + self.fit_regions:
            if s + self.rb > self.ds.n_bins.get(c, 0):
                raise ValueError(f"region {c}:{s} runs past the chromosome end")
        self.strata = self._strata()
        self.sig, self.sig_reason = _signal_module()
        if self.sig is not None:
            self.dnase_route = dnase_route or self.sig.DNASE_ROUTE_DEFAULT
            if self.dnase_route not in self.sig.DNASE_ROUTES:
                raise ValueError(f"dnase_route {self.dnase_route!r} not in {self.sig.DNASE_ROUTES}")
        else:
            self.dnase_route = dnase_route
        self._pa_cache: dict = {}
        self.suff: dict = {}

    # -------------------------------------------------------------- data
    def _strata(self) -> dict:
        names = self.ev.get("strata") or []
        path = self.ev.get("strata_path") or None
        if not names or not path:
            return {}
        from onion_eval.layers import load_strata_regions
        m = load_strata_regions(path, names, self.regions, self.rb)
        return {k: v.reshape(-1) for k, v in m.items()}

    def _tiles(self, regions):
        for ri, (c, s) in enumerate(regions):
            yield ri, [(c, s + k * self.L) for k in range(self.rb // self.L)]

    def _row(self, mid):
        return self.ds.check_row(mid)

    def _encoder_depth(self, row, scale=1.0):
        return self.ds.encoder.c_out(row, scale)

    def _halves_ok(self, row) -> bool:
        return (not self.no_halves) and has_halves(row)

    def _signal_files(self, row):
        """Paths of signal.h5, pr1_signal.h5, pr2_signal.h5, or (None, reason)."""
        if not row["counts_path"].startswith("h5:"):
            return None, f"{row['measurement_id']}: a store: path has no pipeline signal tracks"
        d = Path(row["counts_path"][3:]).parent
        ps = {nm: d / f"{nm}.h5" for nm in ("signal", "pr1_signal", "pr2_signal")}
        miss = [p.name for p in ps.values() if not p.is_file()]
        if miss:
            return None, f"{', '.join(miss)} missing for {row['measurement_id']}"
        return ps, ""

    def _needs_predicted_control(self, job) -> bool:
        if self.sig is None or job["kind"] != "imputation" or job.get("regime") == "control":
            return False
        row = self.ds.rows[job["target"]]
        try:
            route = self.sig.route_of(row["assay_class"] or row["assay"])
        except ValueError:
            return False
        return (route == "chip" and not (row.get("control_measurement_id") or "").strip()
                and self.predict_control and self._signal_files(row)[0] is not None)

    # -------------------------------------------------------------- reference predictors
    def _region_batch(self, mids, chrom, start, which="counts"):
        """A compact batch of whole-region target vectors (one window per measurement)."""
        C = self.C
        Lr = self.rb
        y = np.stack([self.ds.read(m, chrom, start, Lr, which) for m in mids])
        a_idx = [C.TRACK_INDEX[self.ds.rows[m]["assay"]] for m in mids]
        avail = np.zeros((len(mids), C.A), bool)
        c_out = np.zeros((len(mids), C.A, C.K_META), np.float32)
        ycols = np.zeros((len(mids), Lr, C.A), np.int32)
        for b, (m, a) in enumerate(zip(mids, a_idx)):
            avail[b, a] = True
            c_out[b, a] = self._encoder_depth(self.ds.rows[m])
            ycols[b, :, a] = y[b]
        meta = [{"cell": self.ds.rows[m]["cell"], "chrom": chrom, "start_bin": start} for m in mids]
        return {"y_counts": ycols, "y_avail": avail, "c_out": c_out, "meta": meta}

    def _fit_marginal(self, assays):
        from onion_eval.baselines import MarginalNB
        mnb = MarginalNB()
        for a in sorted(assays):
            mids = [m for (c, aa), m in sorted(self.ds.train_base.items()) if aa == a]
            if not mids:
                continue
            batches = [self._region_batch(mids, c, s) for c, s in self.fit_regions]
            part = MarginalNB().fit(batches)
            mnb.params.update(part.params)
        return mnb

    def _pa(self, assay, ri):
        key = (assay, ri)
        if key not in self._pa_cache:
            from onion_eval.baselines import PositionAverage
            if self._pa_cache and next(iter(self._pa_cache))[0] != assay:
                self._pa_cache.clear()
            mids = [m for (c, a), m in sorted(self.ds.train_base.items()) if a == assay]
            pa = PositionAverage()
            if mids:
                c, s = self.regions[ri]
                pa.fit([self._region_batch(mids, c, s)])
            self._pa_cache[key] = pa
        return self._pa_cache[key]

    def _pair_window(self, job, c, s):
        return self.ds.window(c, s, [job["source"]], [job["target"]], input_file="pr1",
                              target_file="pr2")

    def _fit_pair_refs(self, job):
        """Depth-only scaling and the floor of one pair, NB sizes fitted on the fit regions."""
        from onion_eval.baselines import PRED_FLOOR, DepthOnlyScaling, _fit_size
        wins = [self._pair_window(job, c, s) for _, tl in self._tiles(self.fit_regions) for c, s in tl]
        batches = [self.C.collate(wins[i:i + self.bs]) for i in range(0, len(wins), self.bs)]
        dos = DepthOnlyScaling().fit(batches)
        pr1 = np.concatenate([self.ds.read(job["target"], c, s, self.rb, "pr1")
                              for c, s in self.fit_regions]).astype(np.float64)
        pr2 = np.concatenate([self.ds.read(job["target"], c, s, self.rb, "pr2")
                              for c, s in self.fit_regions]).astype(np.float64)
        return dos, _fit_size(np.maximum(pr1, PRED_FLOOR), pr2)

    def _predicted_control(self, batch, t):
        """The model's control mean [B, L] with the control channel asked for at track t's C'
        (a separate forward pass, so the target's own prediction is untouched)."""
        ci = self.C.TRACK_INDEX[CONTROL]
        b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        b2["c_out"][:, ci] = b2["c_out"][:, t]
        b2["y_avail"][:, ci] = True
        b2["y_kind"][:, ci] = 1
        return self.model_pred(b2)["mu"][:, :, ci]

    # -------------------------------------------------------------- one job
    def _job_arrays(self, job, refs):
        """Vectors of one job's target track over every evalset region, in region then tile order."""
        from onion_eval.baselines import PRED_FLOOR
        tgt = self._row(job["target"])
        t = self.C.TRACK_INDEX[tgt["assay"]]
        pair = job["kind"] in ("correction", "replicate")
        halves = (not pair) and self._halves_ok(tgt)
        want_ctl = (not pair) and self.model_pred is not None and self._needs_predicted_control(job)
        if pair:
            dos, floor_n = self._fit_pair_refs(job)
        names = tuple(refs) + ((self.main,) if self.main in REFS and self.main not in refs else ())
        out = defaultdict(list)
        for ri, tiles in self._tiles(self.regions):
            for b0 in range(0, len(tiles), self.bs):
                tl = tiles[b0:b0 + self.bs]
                if pair:
                    wins = [self._pair_window(job, c, s) for c, s in tl]
                else:
                    wins = [self.ds.window(c, s, job["inputs"], [job["target"]]) for c, s in tl]
                batch = self.C.collate(wins)
                preds = {}
                for name in names:
                    if name in ("depth_only_scaling", "pseudoreplicate_floor") and not pair:
                        nanv = np.full(batch["y_counts"].shape, np.nan)
                        preds[name] = {"n": nanv, "mu": nanv}   # defined on pairs only
                    elif name == "marginal_nb":
                        preds[name] = _np_out(self.marg.predict(batch))
                    elif name == "position_average":
                        preds[name] = _np_out(self._pa(tgt["assay"], ri).predict(batch))
                    elif name == "depth_only_scaling":
                        preds[name] = _np_out(dos.predict(batch))
                    elif name == "pseudoreplicate_floor":
                        p1 = np.stack([self.ds.read(job["target"], c, s, self.L, "pr1") for c, s in tl])
                        mu = np.zeros(batch["y_counts"].shape, np.float64)
                        n = np.full(mu.shape, np.nan)
                        mu[:, :, t] = np.maximum(p1, PRED_FLOOR)
                        n[:, :, t] = floor_n
                        preds[name] = {"n": n, "mu": mu}
                preds["main"] = preds[self.main] if self.main in preds else self.model_pred(batch)
                for name, p in preds.items():
                    out[f"{name}.n"].append(p["n"][:, :, t].reshape(-1))
                    out[f"{name}.mu"].append(p["mu"][:, :, t].reshape(-1))
                if want_ctl:
                    out["main.ctl_mu"].append(self._predicted_control(batch, t).reshape(-1))
                out["y"].append(batch["y_counts"][:, :, t].numpy().reshape(-1).astype(np.float64))
                if pair:
                    out["x"].append(batch["x_counts"][:, :, t].numpy().reshape(-1).astype(np.float64))
                    if "log2r" not in out:
                        out["log2r"] = [float(batch["c_out"][0, t, 0] - batch["c_in"][0, t, 0])]
                elif halves:
                    for h in ("pr1", "pr2"):
                        out[h].append(np.concatenate([self.ds.read(job["target"], c, s, self.L, h)
                                                      for c, s in tl]).astype(np.float64))
        res = {k: (np.concatenate(v) if k != "log2r" else v[0]) for k, v in out.items()}
        return res

    # -------------------------------------------------------------- imputation
    def _imputation_group(self, jobs, regime):
        """Per-target records of one (regime, assay) group, and the assay-level shared/specific."""
        from onion_eval import error_budget as EB
        recs, arrays = [], []
        for job in jobs:
            A = self._job_arrays(job, ("marginal_nb", "position_average"))
            n, mu, y = A["main.n"], A["main.mu"], A["y"]
            row = self._row(job["target"])
            halves = self._halves_ok(row)
            r = {"target": job["target"], "assay": job["assay"], "cell": job["cell"],
                 "has_halves": halves}
            no_halves = Mx.Undefined("no pseudoreplicate halves for this target ("
                                     + ("--no-halves mode" if self.no_halves else
                                        "has_pseudoreps != 1 or a store: path") + ")")
            defined = bool(np.isfinite(n).all() and np.isfinite(mu).all() and (mu > 0).all()
                           and (n > 0).all())
            pa_ok = bool(np.isfinite(A["position_average.mu"]).all()
                         and np.isfinite(A["position_average.n"]).all())
            nan = Mx.Undefined("the predictor is undefined on this target (NaN prediction)")
            if self.main == "position_average":
                pa_c = Mx.crps_split(A["position_average.n"], A["position_average.mu"], y)
            elif pa_ok:                # the skill needs only the reference's CRPS (same estimator)
                from candi.bench.distributional import nb_crps_mean
                pa_c = {"crps": nb_crps_mean(A["position_average.n"], A["position_average.mu"], y),
                        "oracle_scaled": float("nan")}
            else:
                pa_c = {k: Mx.Undefined("position average undefined") for k in ("crps", "oracle_scaled")}
            m_c = (pa_c if self.main == "position_average" else Mx.crps_split(n, mu, y)) if defined else {k: nan for k in ("crps", "oracle_scaled", "scale_error")}
            r.update({"crps": m_c["crps"], "oracle_scaled": m_c["oracle_scaled"],
                      "scale_error": m_c["scale_error"], "crps_pa": pa_c["crps"],
                      "skill": (float(pa_c["crps"] - m_c["crps"]) if not (_undef(pa_c["crps"]) or _undef(m_c["crps"]))
                                else Mx.Undefined("CRPS undefined for the model or the position average"))})
            strata = {POOLED: np.ones(y.size, bool), **self.strata}
            r["pit"] = {k: ((pit_curve(n[m], mu[m], y[m]), int(m.sum())) if defined else None)
                        for k, m in strata.items() if m.any()}
            mn, mm = A["marginal_nb.n"], A["marginal_nb.mu"]
            r["pit_marginal"] = ((pit_curve(mn, mm, y), y.size)
                                 if np.isfinite(mn).all() and np.isfinite(mm).all() else None)
            if not halves:
                vr = no_halves
            else:
                vr = (Mx.variance_ratio(n, 0.5 * mu, A["pr1"], A["pr2"], self.strata or None,
                                        depth_ratio_b=1.0)["pooled"] if defined else nan)
            r["variance_ratio"] = vr
            # headline 1 / 2 in counts_log1p: prediction is the NB mean at the target's own C'
            ts = Mx.to_space(y, SPACE)
            h1 = h2 = None
            if halves:
                h1, h2 = (Mx.to_space(A[h], SPACE, rescale=2.0) for h in ("pr1", "pr2"))
                r["N"] = Mx.noise_floor(h1, h2, ts, 2.0)
            else:
                r["N"] = no_halves
            ps = Mx.to_space(mu, SPACE) if defined else None
            a_s = Mx.to_space(A["position_average.mu"], SPACE) if pa_ok else None
            r["r_spec"] = (Mx.r_spec(ts, ps, a_s) if (defined and pa_ok)
                           else Mx.Undefined("prediction or training average undefined"))
            sig = self._signal_target(job, n, mu, A, defined, pa_ok)
            r.update(sig.pop("record"))
            arrays.append({"defined": defined, "truth": ts, "pred": ps, "h1": h1, "h2": h2,
                           "n": n, "mu": mu, **sig})
            recs.append(r)
            self.suff[f"crps/{regime}/{job['target']}"] = np.array(
                [_j(m_c["crps"]) or np.nan, _j(m_c["oracle_scaled"]) or np.nan,
                 _j(pa_c["crps"]) or np.nan, y.size], np.float64)
        # assay-level shared / specific split (headline 1) and error budget, over defined targets;
        # a target without halves joins the shared estimate (it needs no N) but not headline 1
        ok = [i for i, a in enumerate(arrays) if a["defined"]]
        assay = jobs[0]["assay"]
        for i, r in enumerate(recs):
            for k in ("E_sp", "V_sp", "level"):
                r[k] = Mx.Undefined("the predictor is undefined on this target")
            r["budget"] = None
        if ok:
            T = np.stack([arrays[i]["truth"] for i in ok])
            P = np.stack([arrays[i]["pred"] for i in ok])
            cells = np.array([recs[i]["cell"] for i in ok])
            N = None if self.no_halves else np.array([recs[i]["N"] for i in ok], np.float64)
            ss = Mx.shared_specific(T, P, cells, N)
            for j, i in enumerate(ok):
                reason = ss["reason"][j]
                for k in ("E_sp", "V_sp", "level"):
                    v = ss[k][j]
                    recs[i][k] = Mx.Undefined(reason) if (reason and k != "level") else float(v)
            for nm in ("V", "P", "C", "E"):
                for part in ("sh", "sp"):
                    self.suff[f"gram/{regime}/{assay}/{nm}_{part}_by_level"] = ss["by_level"][f"{nm}_{part}"]
            eb = ok if self.no_halves else [i for i in ok if recs[i]["has_halves"]]
            if eb:
                nbl = (None if self.no_halves else
                       EB.noise_share_from_halves(np.stack([arrays[i]["h1"] for i in eb]),
                                                  np.stack([arrays[i]["h2"] for i in eb]), 2.0))
                bud = EB.error_budget(np.stack([arrays[i]["truth"] for i in eb]),
                                      np.stack([arrays[i]["pred"] for i in eb]),
                                      np.array([recs[i]["cell"] for i in eb]), noise_by_level=nbl)
                for i in eb:
                    recs[i]["budget"] = {k: bud[k] for k in BUDGET_KEYS}
            self._signal_group(recs, arrays, ok, regime, assay)
        else:
            for r in recs:
                r["sig_E_sp"] = r["sig_V_sp"] = r["sig_level"] = Mx.Undefined(
                    "the predictor is undefined on this target")
        for r in recs:
            r["recovered"] = recovered_fn(no_floor=self.no_halves)([r])
            for k, v in r["pit"].items():
                if v is not None:
                    self.suff[f"pit/{regime}/{r['target']}/{k}"] = np.append(v[0], v[1])
        return recs

    # -------------------------------------------------------------- signal space (X2)
    def _signal_target(self, job, n, mu, A, defined, pa_ok):
        def undef(reason):
            u = Mx.Undefined(reason)
            return {"record": {"sig_N": u, "sig_r_spec": u, "lambda_source": None,
                               "lambda_reason": reason}}
        if self.sig is None:
            return undef(self.sig_reason)
        row = self._row(job["target"])
        ps_files, why = self._signal_files(row)
        if ps_files is None:
            return undef(why)
        import h5py
        files = {}
        for nm, p in ps_files.items():
            with h5py.File(p, "r") as f:
                files[nm] = np.concatenate([np.asarray(f[c][s:s + self.rb], np.float64)
                                            for c, s in self.regions])
        ts = Mx.to_space(files["signal"], "signal_arcsinh")
        h1 = Mx.to_space(files["pr1_signal"], "signal_arcsinh")
        h2 = Mx.to_space(files["pr2_signal"], "signal_arcsinh")
        kw = dict(ds=self.ds, regions=self.regions, rb=self.rb, dnase_route=self.dnase_route)
        src, lam_reason, ps, pa_reason = None, "", None, ""
        if defined:
            try:
                sp, src = self.sig.signal_from_nb(n, mu, row, predicted_control=A.get("main.ctl_mu"),
                                                  **kw)
                ps = Mx.to_space(np.asarray(sp, np.float64), "signal_arcsinh")
            except (ValueError, KeyError) as e:
                lam_reason = f"no signal from the prediction: {e}"
        else:
            lam_reason = "the predictor is undefined on this target"
        a_s = None
        if pa_ok:
            try:
                sa, _ = self.sig.signal_from_nb(A["position_average.n"], A["position_average.mu"],
                                                row, predicted_control=None, **kw)
                a_s = Mx.to_space(np.asarray(sa, np.float64), "signal_arcsinh")
            except (ValueError, KeyError) as e:
                a_s, pa_reason = None, f"no signal from the position average: {e}"
        N = Mx.noise_floor(h1, h2, ts, 2.0)
        if ps is not None and a_s is not None:
            rs = Mx.r_spec(ts, ps, a_s)
        else:
            rs = Mx.Undefined(lam_reason or (pa_reason if pa_ok else "position average undefined"))
        return {"record": {"sig_N": N, "sig_r_spec": rs, "lambda_source": src,
                           "lambda_reason": lam_reason},
                "sig_truth": ts, "sig_pred": ps}

    def _signal_group(self, recs, arrays, ok, regime, assay):
        for r in recs:
            r["sig_E_sp"] = r["sig_V_sp"] = r["sig_level"] = Mx.Undefined(
                "the predictor is undefined on this target")
        if self.sig is None or any(arrays[i].get("sig_pred") is None for i in ok):
            reasons = sorted({recs[i].get("lambda_reason") or "" for i in ok} - {""})
            reason = self.sig_reason or ("; ".join(reasons) if reasons
                                         else "signal prediction or truth missing")
            for r in recs:
                r["sig_E_sp"] = r["sig_V_sp"] = r["sig_level"] = Mx.Undefined(reason)
            return
        T = np.stack([arrays[i]["sig_truth"] for i in ok])
        P = np.stack([arrays[i]["sig_pred"] for i in ok])
        ss = Mx.shared_specific(T, P, np.array([recs[i]["cell"] for i in ok]),
                                np.array([recs[i]["sig_N"] for i in ok], np.float64))
        for j, i in enumerate(ok):
            for k in ("E_sp", "V_sp"):
                recs[i][f"sig_{k}"] = (Mx.Undefined(ss["reason"][j]) if ss["reason"][j]
                                       else float(ss[k][j]))
            recs[i]["sig_level"] = float(ss["level"][j])      # finding 15: signal-space level

    # -------------------------------------------------------------- pairs
    def _pair_record(self, job):
        A = self._job_arrays(job, ("depth_only_scaling", "pseudoreplicate_floor"))
        y = A["y"]
        out = {k: job.get(k) for k in ("pair_id", "assay", "cell", "knob", "never_trained",
                                       "subset_pair", "rung", "source", "target")}
        for nm, key in (("main", "crps_model"), ("depth_only_scaling", "crps_depth_only"),
                        ("pseudoreplicate_floor", "crps_floor")):
            n, mu = A[f"{nm}.n"], A[f"{nm}.mu"]
            if np.isfinite(n).all() and np.isfinite(mu).all():
                if nm == "main":       # the oracle split is quoted for the model's CRPS only
                    c = Mx.crps_split(n, mu, y)
                    out[key], out[key + "_oracle_scaled"] = c["crps"], c["oracle_scaled"]
                else:                  # same estimator as crps_split's crps (closed form)
                    from candi.bench.distributional import nb_crps_mean
                    out[key] = nb_crps_mean(n, mu, y)
                    out[key + "_oracle_scaled"] = None
            else:
                out[key] = out[key + "_oracle_scaled"] = Mx.Undefined(f"{nm} undefined on this pair")
        mu = A["main.mu"]
        out["pred_total"] = float(mu.sum()) if np.isfinite(mu).all() else float("nan")
        out["source_total"] = float(A["x"].sum())
        out["log2_depth_ratio"] = A["log2r"]
        self.suff[f"pair/{job.get('pair_id') or job['source'] + '>' + job['target']}"] = np.array(
            [_j(out["crps_model"]) or np.nan, _j(out["crps_depth_only"]) or np.nan,
             _j(out["crps_floor"]) or np.nan, y.size], np.float64)
        return out

    # -------------------------------------------------------------- guardrail
    def _copy_leak(self):
        if self.model is None:
            return {"hidden_input_max_abs_change": None, "unavailable_input_max_abs_change": None,
                    "same_view_copy_gain": None, "copy_leak_pass": None,
                    "copy_leak_spread_ok": None, "part1_wiring_pass": None,
                    "reason": "not applicable: a training-free baseline reads no hidden input"}
        from onion_eval.guardrail import GAIN_THRESHOLD, copy_leak
        per_seed, n_win = [], 0
        for s in self.guardrail_seeds:
            ds = self._OnionDataset(self.tables, self.split, self.cfg, seed=s)
            items = [it for it in ds.evalset_items() if it["task"] == "reconstruction"]
            items = items[: self.guardrail_windows]
            wins = [ds.item_window(it) for it in items]
            n_win = len(wins)
            batches = [self.C.collate(wins[i:i + self.bs]) for i in range(0, len(wins), self.bs)]
            per_seed.append(copy_leak(self.model_pred.model, batches, seed=s))
        if not per_seed:
            return {"copy_leak_pass": None, "copy_leak_spread_ok": None, "part1_wiring_pass": None,
                    "reason": "no mask seed"}
        rel = [_j(r["same_view_copy_gain_relative"]) for r in per_seed]
        fin = [v for v in rel if v is not None]
        spread = (max(fin) - min(fin)) if len(fin) >= 2 else None
        lim = SPREAD_MAX_FRACTION * GAIN_THRESHOLD
        spread_ok = bool(spread is not None and len(fin) == len(rel) and spread <= lim)

        def mx(k):
            v = [_j(r[k]) for r in per_seed]
            return None if any(x is None for x in v) else max(v)
        return {
            "hidden_input_max_abs_change": mx("hidden_input_max_abs_change"),
            "unavailable_input_max_abs_change": mx("unavailable_input_max_abs_change"),
            "same_view_copy_gain": mx("same_view_copy_gain"),
            "same_view_copy_gain_relative": mx("same_view_copy_gain_relative"),
            "copy_leak_pass": all(r["copy_leak_pass"] is True for r in per_seed),
            "part1_wiring_pass": all(r["part1_wiring_pass"] is True for r in per_seed),
            "part2_j_invariance_pass": all(r["part2_j_invariance_pass"] is True for r in per_seed),
            "copy_leak_pass_by_mask_seed": [bool(r["copy_leak_pass"]) for r in per_seed],
            "hidden_input_max_abs_change_by_mask_seed": [_j(r["hidden_input_max_abs_change"])
                                                         for r in per_seed],
            "unavailable_input_max_abs_change_by_mask_seed": [
                _j(r["unavailable_input_max_abs_change"]) for r in per_seed],
            "same_view_copy_gain_relative_by_mask_seed": rel,
            "same_view_copy_gain_relative_spread": spread,
            "copy_leak_spread_ok": spread_ok,
            "spread_limit_relative": lim, "gain_threshold_relative": GAIN_THRESHOLD,
            "mask_seeds": list(self.guardrail_seeds), "n_windows_per_seed": n_win,
            "reason": "" if spread_ok else (
                f"the relative gain moved {spread} across mask seeds (limit {lim:g}); score with "
                f"more --guardrail-windows" if spread is not None else
                "the relative gain is undefined on a mask seed, or fewer than 2 mask seeds"),
        }

    # -------------------------------------------------------------- run
    def jobs(self):
        meas = self.ds.rows
        imp = [dict(t, kind="imputation") for t in self.ev["imputation_targets"]
               if t["split"] == self.split]
        ctl = [dict(t, kind="imputation", regime="control")
               for t in self.ev.get("control_targets", []) if t["split"] == self.split]
        for t in imp + ctl:
            t.setdefault("assay", meas[t["target"]]["assay"])
            t.setdefault("cell", meas[t["target"]]["cell"])
        bad_ctl = [t["target"] for t in imp if t["assay"] == CONTROL]
        if bad_ctl:
            raise ValueError(f"control tracks listed as imputation targets: {bad_ctl[:3]} "
                             f"(rebuild the eval set)")
        cor = [dict(p, kind="correction") for p in self.ev["correction_pairs"]
               if p["split"] == self.split]
        for p in cor:
            p.setdefault("assay", meas[p["target"]]["assay"])
            p.setdefault("cell", meas[p["target"]]["cell"])
        rep = []
        want = set(self.ev.get("replicate_pairs") or [])
        if want:
            from onion_eval.evalset import read_tsv
            for p in read_tsv(self.tables / "pairs.tsv"):
                if (p["pair_id"] in want and p["split"] == self.split and p["rung"] in REPLICATE_RUNGS
                        and p["a"] != p["b"]
                        and all(has_halves(meas.get(m, {})) for m in (p["a"], p["b"]))):
                    rep.append({"kind": "replicate", "source": p["a"], "target": p["b"],
                                "pair_id": p["pair_id"], "rung": p["rung"], "assay": p["assay"],
                                "cell": p["cell"], "never_trained": int(p["never_trained"] or 0),
                                "subset_pair": int(p["subset_pair"] or 0),
                                "knob": p.get("knobs_differing", "")})
        if self.no_halves:          # every pair reads halves
            cor, rep = [], []
        for j in imp + ctl + cor + rep:
            for m in [j["target"]] + j.get("inputs", []) + ([j["source"]] if "source" in j else []):
                if m not in meas:
                    raise KeyError(f"measurement {m!r} not in measurements.tsv")
                if meas[m]["in_eic"] == "blind" and self.split != "blind":
                    raise PermissionError(f"{m} is a blind track; blind rows are scored only on "
                                          f"--split blind --allow-blind")
                self.ds.check_row(m)
        if not self.predict_control:
            ctl = []
        return imp, cor, rep, ctl

    def run(self) -> dict:
        t0 = time.time()
        imp, cor, rep, ctl = self.jobs()
        sc = self.ev.get("scorable", {})
        sc_imp, sc_cor = sc.get("imputation", []), sc.get("correction", [])
        self.marg = self._fit_marginal({j["assay"] for j in imp + ctl})
        scores = {}
        recs_by_regime = {}
        for regime in REGIMES + ("control",):
            groups = defaultdict(list)
            for j in (ctl if regime == "control" else imp):
                if j["regime"] == regime:
                    groups[j["assay"]].append(j)
            recs = []
            for a in sorted(groups):
                self.log(f"score: {regime} {a}: {len(groups[a])} targets")
                recs += self._imputation_group(groups[a], regime)
            self._pa_cache.clear()
            recs_by_regime[regime] = recs
        main_recs = recs_by_regime["leave_one_assay_out"]
        scores.update(self._imputation_metrics(main_recs, sc_imp, require_all=True))
        scores["accessibility_only"] = self._imputation_metrics(
            recs_by_regime["accessibility_only"], sc_imp)
        scores["accessibility_only"]["regime"] = "accessibility_only"
        if ctl:
            pc = self._imputation_metrics(recs_by_regime["control"], [CONTROL])
            pc["regime"] = "control"
            pc["n_targets"] = len(recs_by_regime["control"])
            scores["predicted_control"] = pc
        else:
            scores["predicted_control"] = {"n_targets": 0, "reason": (
                "predict_control is off or the predictor is a training-free baseline"
                if not self.predict_control else "no control target in this split")}
        self.log(f"score: {len(cor)} correction pairs, {len(rep)} replicate pairs")
        cor_recs = [self._pair_record(j) for j in cor]
        rep_recs = [self._pair_record(j) for j in rep]
        scores.update(self._correction_metrics(cor_recs, sc_cor))
        scores["replicate_pair_gap_closed_vs_depth_only"] = {"per_rung": {
            rung: metric([r for r in rep_recs if r["rung"] == rung], _gap_fn, set(
                (sc.get("layers") or {}).get(rung, [])), self.n_boot, self.seed)
            for rung in sorted({r["rung"] for r in rep_recs})}}
        scores["copy_leak"] = self._copy_leak()
        if self.no_halves:
            imp_t = Mx.Undefined("--no-halves mode: headline 1 has no noise floor and is not a "
                                 "selection term")
        else:
            imp_t = scores["specific_variance_recovered_above_noise"][SPACE]["macro"]
        cor_m = scores["correction_gap_closed_vs_depth_only_never_trained"]
        cor_t = cor_m["macro"]
        unc, unc_den = self._uncertainty(main_recs, sc_imp)
        comp = composite(imp_t, cor_t, unc)
        den_rel = [v for v in cor_m["denominator_relative_by_assay"].values() if v is not None]
        scores["selection"] = {
            "space": SPACE, "imputation_gap_closed": _j(imp_t),
            "correction_gap_closed": _j(cor_t), "uncertainty_gap_closed": _j(unc),
            "composite": _j(comp), "reason": _why(comp) if _undef(comp) else "",
            "imputation_reason": _why(imp_t) if _undef(imp_t) else "",
            "uncertainty_reason": _why(unc) if _undef(unc) else "",
            "denominators": {
                "uncertainty_marginal_pit_deviation": _j(unc_den),
                "correction_baseline_minus_floor_relative_min": (min(den_rel) if den_rel and len(
                    den_rel) == len(cor_m["denominator_relative_by_assay"]) else None),
                "correction_baseline_minus_floor_relative_by_assay":
                    cor_m["denominator_relative_by_assay"]}}
        n_wo = sum(1 for r in main_recs if not r["has_halves"])
        scores.update({"n_targets": len(main_recs), "n_pairs": len(cor_recs),
                       "n_targets_without_halves": n_wo,
                       "n_replicate_pairs": len(rep_recs), "regime": "leave_one_assay_out",
                       "split": self.split, "scorable": sc, "predictor": self.main,
                       "noise_floor": NOISE_FLOOR_NONE if self.no_halves else NOISE_FLOOR_HALVES,
                       "evalset": str(self.evalset_path), "evalset_sha256": sha256_file(self.evalset_path),
                       "tables": str(self.tables), "tables_sha256": self.tables_sha256,
                       "run_tables_sha256": self.run_tables_sha256,
                       "dnase_route": self.dnase_route, "baseline_fit_on": self.fit_on,
                       "n_regions": len(self.regions), "region_bins": self.rb,
                       "context_bins": self.L, "wall_seconds": round(time.time() - t0, 1)})
        return scores

    # -------------------------------------------------------------- metric assembly
    def _imputation_metrics(self, recs, scorable, require_all=False) -> dict:
        nb, sd = self.n_boot, self.seed
        ra = require_all
        wo = sorted(r["target"] for r in recs if not r["has_halves"])
        if self.no_halves:
            h1 = metric(recs, None, scorable, undefined_reason=(
                f"--no-halves mode: no pseudoreplicate noise floor; headline 1 without a floor is "
                f"under {PILOT_H1_KEY} and is never a selection input"))
            pilot = metric(recs, recovered_fn(no_floor=True), scorable, nb, sd, require_all=ra)
        else:
            h1 = metric(recs, recovered_fn(), scorable, nb, sd, require_all=ra)
        h1["n_targets_without_halves"] = len(wo)
        h1["targets_without_halves"] = wo
        h1["per_target"] = {r["target"]: {"assay": r["assay"], "cell": r["cell"], "level": _j(r["level"]),
                                          "E_sp": _j(r["E_sp"]), "V_sp": _j(r["V_sp"]), "N": _j(r["N"]),
                                          "has_halves": r["has_halves"],
                                          "recovered": _j(r["recovered"])} for r in recs}
        h1s = metric(recs, recovered_fn("sig_"), scorable, nb, sd, require_all=ra)
        h1s["per_target"] = {r["target"]: {"assay": r["assay"], "cell": r["cell"],
                                           "level": _j(r["sig_level"]), "N": _j(r["sig_N"]),
                                           "lambda_source": r.get("lambda_source"),
                                           "lambda_reason": r.get("lambda_reason", "")}
                             for r in recs}
        h1s["dnase_route"] = self.dnase_route

        def pit_fn(key):
            def fn(rs):
                cur = [r["pit"].get(key) for r in rs]
                if any(c is None for c in cur) and key == POOLED:
                    return Mx.Undefined("the predictor is undefined on a target")
                return pooled_pit_deviation([c for c in cur if c is not None])
            return fn

        def worst_fn(rs):
            if not self.strata:
                return Mx.Undefined("no region strata in the evalset")
            vals = [pit_fn(k)(rs) for k in self.strata]
            vals = [v for v in vals if not _undef(v)]
            return max(vals) if vals else Mx.Undefined("no stratum with bins")

        budget = {}
        per_assay_budget = {}
        by = defaultdict(list)
        for r in recs:
            if r.get("budget") is not None:
                by[r["assay"]].append(r["budget"])
        for a, bs in by.items():
            per_assay_budget[a] = {k: _j(bs[0][k]) for k in BUDGET_KEYS}
        for k in BUDGET_KEYS:
            vals = [per_assay_budget[a][k] for a in scorable if a in per_assay_budget]
            budget[k] = (float(np.mean(vals)) if vals and all(v is not None for v in vals)
                         and len(vals) == len([a for a in scorable if a in {r["assay"] for r in recs}])
                         else None)
        budget["per_assay"] = per_assay_budget
        out = {
            "specific_variance_recovered_above_noise": {SPACE: h1, "signal_arcsinh": h1s},
            "r_spec": {SPACE: metric(recs, _mean_of("r_spec"), scorable, nb, sd, require_all=ra),
                       "signal_arcsinh": metric(recs, _mean_of("sig_r_spec"), scorable, nb, sd,
                                                require_all=ra)},
            "count_crps_skill_vs_position_average": metric(recs, _mean_of("skill"), scorable, nb, sd,
                                                           require_all=ra),
            "count_crps": metric(recs, _mean_of("crps"), scorable, nb, sd, require_all=ra),
            "count_crps_oracle_scaled": metric(recs, _mean_of("oracle_scaled"), scorable, nb, sd,
                                               require_all=ra),
            "count_crps_scale_error": metric(recs, _mean_of("scale_error"), scorable, nb, sd,
                                             require_all=ra),
            "pit_deviation_pooled": metric(recs, pit_fn(POOLED), scorable, nb, sd, require_all=ra),
            "pit_deviation_worst_stratum": metric(recs, worst_fn, scorable, nb, sd, require_all=ra),
            "predicted_over_replicate_variance": metric(recs, _mean_of("variance_ratio"), scorable,
                                                        nb, sd, require_all=ra),
            "error_budget": budget,
        }
        if self.no_halves:
            pilot["per_target"] = h1["per_target"]
            out[PILOT_H1_KEY] = pilot
        return out

    def _uncertainty(self, recs, scorable):
        """(uncertainty term, its denominator = mean marginal-NB PIT deviation over the assays)."""
        by = defaultdict(list)
        for r in recs:
            if r["assay"] in scorable:
                by[r["assay"]].append(r)
        if not by:
            return Mx.Undefined("no scorable assay with a target"), None
        missing = sorted(set(scorable) - set(by))
        if missing:
            return Mx.Undefined("scorable assay with no target: " + ", ".join(missing)), None
        arm, marg = [], []
        for a, rs in sorted(by.items()):
            if any(r["pit"].get(POOLED) is None for r in rs):
                return Mx.Undefined(f"PIT undefined for the arm on {a}"), None
            if any(r["pit_marginal"] is None for r in rs):
                return Mx.Undefined(f"marginal NB undefined on {a}"), None
            arm.append(pooled_pit_deviation([r["pit"][POOLED] for r in rs]))
            marg.append(pooled_pit_deviation([r["pit_marginal"] for r in rs]))
        return Mx.uncertainty_gap_closed(arm, marg), float(np.mean(marg))

    def _correction_metrics(self, recs, scorable) -> dict:
        nb, sd = self.n_boot, self.seed
        head = [r for r in recs if r["never_trained"] == 1 and r["subset_pair"] == 0]
        sub = [r for r in recs if r["subset_pair"] == 1]        # pure depth thinning, any pair
        m = metric(head, _gap_fn, scorable, nb, sd, require_all=True)
        knobs = defaultdict(list)
        for r in head:
            knobs[r.get("knob") or ""].append(r)
        m["per_knob"] = {k: _j(_gap_fn(rs)) for k, rs in sorted(knobs.items())}
        m["n_pairs"] = len(head)
        den = {}
        for a in sorted(scorable):
            rs = [r for r in head if r["assay"] == a]
            b = [r["crps_depth_only"] for r in rs]
            f = [r["crps_floor"] for r in rs]
            if rs and not any(_undef(x) for x in b + f) and np.mean(b) > 0:
                den[a] = float((np.mean(b) - np.mean(f)) / np.mean(b))
            else:
                den[a] = None
        m["denominator_relative_by_assay"] = den

        def law(rs):
            bad = [r for r in rs if any(_undef(r[k]) for k in ("pred_total", "source_total",
                                                               "log2_depth_ratio"))]
            if bad:
                return Mx.Undefined(f"{len(bad)} of {len(rs)} pairs have an undefined total")
            return Mx.depth_law_error([r["pred_total"] for r in rs], [r["source_total"] for r in rs],
                                      [r["log2_depth_ratio"] for r in rs])
        return {
            "correction_gap_closed_vs_depth_only_never_trained": m,
            "correction_crps_model": metric(head, _mean_of("crps_model"), scorable, nb, sd),
            "correction_crps_model_oracle_scaled": metric(head, _mean_of("crps_model_oracle_scaled"),
                                                          scorable, nb, sd),
            "correction_crps_depth_only": metric(head, _mean_of("crps_depth_only"), scorable, nb, sd),
            "correction_crps_pseudoreplicate_floor": metric(head, _mean_of("crps_floor"), scorable, nb, sd),
            "subset_pairs_gap_closed": metric(sub, _gap_fn, scorable, nb, sd),
            "depth_law_abs_log2_error_never_trained": metric(head, law, scorable, nb, sd),
        }


def _gap_fn(rs):
    keys = ("crps_model", "crps_depth_only", "crps_floor")
    bad = [r for r in rs if any(_undef(r[k]) for k in keys)]
    if bad:
        k = next(k for k in keys if _undef(bad[0][k]))
        return Mx.Undefined(f"{len(bad)} of {len(rs)} pairs undefined: {_why(bad[0][k])}")
    return Mx.gap_closed([r["crps_model"] for r in rs], [r["crps_depth_only"] for r in rs],
                         [r["crps_floor"] for r in rs])


# ------------------------------------------------------------------------------------------ blind ledger

def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ledger_rows(path) -> list[dict]:
    if not Path(path).is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter="\t")]


def ledger_append(path, row: dict) -> None:
    """Append-only: one line per event, never rewritten."""
    path = Path(path)
    new = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(LEDGER_COLUMNS), delimiter="\t",
                           lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEDGER_COLUMNS})
        fh.flush()
        os.fsync(fh.fileno())


# ------------------------------------------------------------------------------------------ entry

def write_outputs(scores: dict, suff: dict, out) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "scores.json.tmp"
    tmp.write_text(json.dumps(scores, indent=1, sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    np.savez_compressed(out / "suffstats.npz", **{k: np.asarray(v) for k, v in suff.items()})
    tmp.replace(out / "scores.json")
    (out / SCORED).write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n", encoding="utf-8")
    return out / "scores.json"


def run_tables_sha256(run) -> str:
    """tables_sha256 recorded in the run's checkpoint (ckpt.pt; T1's pinned dict)."""
    p = Path(run) / "ckpt.pt"
    if not p.is_file():
        raise ScoreRefused(f"{p}: no checkpoint; cannot check the run's tables")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if not ck.get("tables_sha256"):
        raise ScoreRefused(f"{p}: no tables_sha256 in the checkpoint")
    return str(ck["tables_sha256"])


def score(evalset, out, *, run=None, baseline=None, model=None, config=None,
          repeat_blind_reason=None, **kw) -> dict:
    """Score a run dir, an in-memory model, or a baseline name; writes the output files."""
    if sum(x is not None for x in (run, baseline, model)) != 1:
        raise ValueError("give exactly one of run, baseline, model")
    split = kw.get("split", "val")
    out = Path(out)
    if (split == "blind") != (out.name == "blind"):
        raise ScoreRefused(f"--out {out}: a blind pass writes into a dir named 'blind' "
                           f"(<seed dir>/blind), and only a blind pass does")
    run_sha = kw.pop("run_tables_sha256", None)
    if run is not None:
        run_sha = run_tables_sha256(run)
        from minicandi.registry import load_run          # lazy: chunk T1 owns it
        model, config = load_run(run)
    main = baseline if baseline is not None else "model"
    if baseline is not None and baseline not in REFS:
        raise ValueError(f"unknown baseline {baseline!r}; one of {REFS}")
    sc = Scorer(evalset, main=main, model=model, config=config, run_tables_sha256=run_sha, **kw)
    ledger, entry = None, None
    if split == "blind":
        ledger = sc.evalset_path.parent / BLIND_LEDGER
        key = (str(Path(run).resolve()) if run is not None else
               f"baseline:{baseline}" if baseline is not None else f"model:{out.resolve()}")
        esha = sha256_file(sc.evalset_path)
        prior = [r for r in ledger_rows(ledger) if r["run"] == key and r["evalset_sha256"] == esha]
        reason = (repeat_blind_reason or "").strip()
        if prior and not reason:
            raise ScoreRefused(f"{key} already had {len(prior)} blind-pass ledger rows for this eval "
                               f"set ({ledger}); a repeat needs --repeat-blind-reason \"<text>\"")
        entry = {"run": key, "evalset_sha256": esha, "tables_sha256": sc.tables_sha256,
                 "git_sha": git_sha(), "out": str(out.resolve()),
                 "repeat_reason": reason if prior else ""}
        ledger_append(ledger, dict(entry, time=_utc(), status="started"))
    scores = sc.run()
    if run is not None:
        scores["run"] = str(Path(run).resolve())
    if ledger is not None:
        scores["blind_ledger"] = str(ledger)
    write_outputs(scores, sc.suff, out)
    if ledger is not None:
        ledger_append(ledger, dict(entry, time=_utc(), status="scored"))
    return scores


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--baseline", choices=REFS)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val", choices=("val", "blind"))
    ap.add_argument("--allow-blind", action="store_true")
    ap.add_argument("--repeat-blind-reason", default=None)
    ap.add_argument("--no-halves", action="store_true",
                    help="pilot tables without pseudoreplicate halves: headline 1 without a noise "
                         "floor, under a key the selection rule refuses")
    ap.add_argument("--dnase-route", default=None,
                    choices=("hotspot2_density", "macs2_no_control"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--guardrail-windows", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if (Path(a.out) / SCORED).is_file() and not a.force:
        print(f"skip: {a.out}/{SCORED} exists")
        return 0
    try:
        s = score(a.evalset, a.out, run=a.run, baseline=a.baseline, split=a.split,
                  allow_blind=a.allow_blind, device=a.device, n_boot=a.n_boot,
                  guardrail_windows=a.guardrail_windows, no_halves=a.no_halves,
                  dnase_route=a.dnase_route, repeat_blind_reason=a.repeat_blind_reason)
    except (PermissionError, ScoreRefused) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    sel = s["selection"]
    print(f"scores -> {a.out}/scores.json; composite {sel['composite']} (imputation "
          f"{sel['imputation_gap_closed']}, correction {sel['correction_gap_closed']}, uncertainty "
          f"{sel['uncertainty_gap_closed']}); copy_leak_pass {s['copy_leak'].get('copy_leak_pass')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
