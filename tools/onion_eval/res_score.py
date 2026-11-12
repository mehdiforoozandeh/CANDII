"""Score the resolution arms at a common bin size (Amendment 3, section C, chunk RM).

    python tools/onion_eval/res_score.py predict --run <run dir> --tables <eic_res_v1> --evalset <json>
        [--device cpu] [--infer-batch 8] [--force]
    python tools/onion_eval/res_score.py table --run <run dir> --at 25|50|125|250 --out <json>
        [--n-boot 200] [--bound both|independent|comonotone]

predict. Once per run. For every validation imputation target of the headline regime
(`leave_one_assay_out`) it runs the model on the run's own token grid (`ResDataset`, the region
store) over each evalset region, tiled from the region's first base in whole windows of
`context_bins * res_bp` bp, and keeps the target track's (mu, n) at the arm's output bin
(`res_bp`, or 5 bp under `res_out_sub = 5`, from `mu_sub`, `n_sub`). It writes
`<run dir>/res_pred/<target>.npz` = {mu_sum, v_ind, s_com, bin_bp}: per region [n_regions, n_bins],
at the cache bin = the coarser of (output bin, 25 bp), the additive statistics of the output bins
inside each cache bin, mu_sum = sum mu_i, v_ind = sum mu_i^2 / n_i, s_com = sum mu_i / sqrt(n_i)
(mu and n clamped below as `losses.pool_nb` does). `res_pred/INFO.json` records the run config,
the tables and eval set with their sha256, the output and cache bins and the bins per region. An
existing INFO.json skips the run unless --force.

table. Reads only the cache and the truth; never runs a model. `--at` b must be a multiple of the
arm's output bin and of its cache bin, else the arm is refused (exit 2): a 50 bp arm has no 125 bp
table, a 125 bp arm no 25 bp table. Without `res_pred/INFO.json` it fails (exit 1). Span and Haar
depth (section C): at 25 bp the whole region, with `score.py`'s own Haar depth (J = 14 on
16,384-bin regions); above 25 bp the first `2**J * b` bp of each region with
J = floor(log2(region bp / b)), i.e. J = 13, 11, 10 at 50, 125, 250 bp on 409,600 bp regions
(125 and 250 bp use the first 256,000 bp). Every vector is pooled to b exactly: the truth, its
pseudoreplicate halves and the strata masks (a b-bin is in a stratum when all its 25 bp bins are)
by block sums; the model by block sums of the cache, with the pooled NB size
  independent  n = (sum mu)^2 / sum(mu^2 / n)              (the moment-matched size)
  comonotone   n = (sum mu)^2 / (sum mu / sqrt(n))^2        (the perfectly correlated bound)
one table per bound; the reference predictors of `score.py` (marginal NB and position average, fitted
at 25 bp as `score.py` fits them) are pooled with `losses.pool_nb` (the independent size).

The metrics are `score.py`'s own code on the pooled vectors (`CacheScorer` subclasses its `Scorer`
and replaces only the per-target vectors; `metrics.py` computes every number): per bound
`imputation_gap_closed` (headline 1 in counts_log1p, macro over the scorable assays),
`uncertainty_gap_closed`, `pit_deviation_pooled`, `count_crps`, `count_crps_oracle_scaled`,
`count_crps_scale_error` (macros), each metric's per-assay values, and the reasons of undefined
terms. The eval set's correction and replicate pairs are not scored (the region tables carry no
designed pair, section C), nor the signal space, the error budget and the copy-leak guardrail. At
25 bp a run of a 25 bp arm gives `score.py`'s values on the same run within 1e-6
(tests/test_minicandi_resolution.py). `score.py` is not edited.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import math
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi import contract as C  # noqa: E402
from minicandi.components import res_bp_of, res_subs_of, sub_to_fine  # noqa: E402
from minicandi.losses import pool_nb  # noqa: E402
from onion_eval import metrics as Mx  # noqa: E402
from onion_eval import score as S  # noqa: E402

CACHE = "res_pred"
INFO = "INFO.json"
SIZES = (25, 50, 125, 250)
BOUNDS = ("independent", "comonotone")
REGIME = "leave_one_assay_out"
TABLE_KEYS = ("count_crps", "count_crps_oracle_scaled", "count_crps_scale_error",
              "pit_deviation_pooled")


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_bin_bp(config: dict) -> int:
    """The bin of the arm's output: 5 bp under res_out_sub = 5, else res_bp."""
    res = res_bp_of(config)
    return C.RES // res_subs_of(config)[1] if res_subs_of(config)[1] > 1 else res


def span_of(at: int, region_bins: int) -> tuple[int, int | None]:
    """(25 bp bins of each region that the table at `at` bp uses, Haar depth J or None for
    score.py's own). At 25 bp the whole region; above, the first 2**J bins of `at` bp."""
    if at == C.RES:
        return int(region_bins), None
    J = int(math.floor(math.log2(int(region_bins) * C.RES // int(at))))
    return (2 ** J) * int(at) // C.RES, J


# --------------------------------------------------------------------------------------- predict

def _headline_jobs(ev: dict) -> list[dict]:
    return [t for t in ev.get("imputation_targets", [])
            if t.get("split") == "val" and t.get("regime") == REGIME]


def predict(run, tables, evalset, *, model=None, config=None, device="cpu", infer_batch=8,
            force=False, log=print) -> Path:
    """Write `<run>/res_pred/` (module docstring); returns its INFO.json path."""
    from minicandi.dataset import tables_sha256
    from minicandi.res_dataset import ResDataset
    run, tables, evalset = Path(run), Path(tables), Path(evalset)
    cache = run / CACHE
    if (cache / INFO).is_file() and not force:
        log(f"skip: {cache / INFO} exists (--force to predict again)")
        return cache / INFO
    if model is None:
        from minicandi.registry import load_run
        model, config = load_run(run, map_location=device)
    inner = getattr(model, "inner", model)          # a ResAdapter's fine model
    inner = inner.to(device).eval()
    cfg = dict(config)
    out_sub = res_subs_of(cfg)[1]
    out_bin = output_bin_bp(cfg)
    cache_bin = max(out_bin, C.RES)
    g = cache_bin // out_bin
    ev = json.loads(evalset.read_text(encoding="utf-8"))
    rb = int(ev.get("region_bins", 2 ** 14))
    region_bp = rb * C.RES
    ds = ResDataset(tables, "val", {**cfg, "evalset": str(evalset.resolve()), "region_bins": rb})
    wbp = ds.window_bp
    nw = region_bp // wbp
    if nw < 1:
        raise ValueError(f"a window of {wbp} bp does not fit in a {region_bp} bp eval region")
    n_cache = (nw * wbp // out_bin) // g
    regions = [(c, int(s)) for c, s in ev["regions"]]
    cache.mkdir(parents=True, exist_ok=True)
    jobs = _headline_jobs(ev)
    for j, job in enumerate(jobs):
        t = C.TRACK_INDEX[ds.rows[job["target"]]["assay"]]
        mus, ns = [], []
        for chrom, s25 in regions:
            starts = [s25 * C.RES + k * wbp for k in range(nw)]
            m_r, n_r = [], []
            for i in range(0, nw, int(infer_batch)):
                wins = [ds.window(chrom, b, job["inputs"], [job["target"]])
                        for b in starts[i:i + int(infer_batch)]]
                batch = C.collate(wins)
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                with torch.no_grad():
                    o = inner(batch)
                if out_sub > 1:
                    mu, n = sub_to_fine(o["mu_sub"])[..., t], sub_to_fine(o["n_sub"])[..., t]
                else:
                    mu, n = o["mu"][..., t], o["n"][..., t]
                m_r.append(mu.detach().cpu().double().reshape(-1))
                n_r.append(n.detach().cpu().double().reshape(-1))
            mus.append(torch.cat(m_r)[: n_cache * g])
            ns.append(torch.cat(n_r)[: n_cache * g])
        mu = torch.stack(mus).clamp_min(C.MU_MIN).reshape(len(regions), n_cache, g)
        n = torch.stack(ns).clamp_min(C.N_MIN).reshape(len(regions), n_cache, g)
        np.savez(cache / f"{job['target']}.npz",
                 mu_sum=mu.sum(-1).numpy(), v_ind=(mu * mu / n).sum(-1).numpy(),
                 s_com=(mu / n.sqrt()).sum(-1).numpy(), bin_bp=np.int64(cache_bin))
        log(f"predict: {j + 1}/{len(jobs)} {job['target']}")
    info = {"run": str(run.resolve()), "config": cfg, "tables": str(tables.resolve()),
            "tables_sha256": tables_sha256(tables), "evalset": str(evalset.resolve()),
            "evalset_sha256": _sha256(evalset), "regime": REGIME,
            "targets": [j["target"] for j in jobs], "regions": [list(r) for r in regions],
            "region_bins": rb, "res_bp": res_bp_of(cfg), "res_in_sub": res_subs_of(cfg)[0],
            "res_out_sub": out_sub, "output_bin_bp": out_bin, "cache_bin_bp": cache_bin,
            "n_cache_bins": n_cache, "window_bp": wbp, "windows_per_region": nw,
            "git_sha": S.git_sha()}
    (cache / INFO).write_text(json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return cache / INFO


# ----------------------------------------------------------------------------------------- table

class _NoModel(C.MiniCandi):
    """Stands in for the model inside `Scorer`; its output is replaced by the cache."""

    def forward(self, batch: dict) -> dict:
        one = torch.ones(batch["y_counts"].shape, dtype=torch.float32)
        return {"n": one, "mu": one}


def _block_sum(v: np.ndarray, nreg: int, rb: int, span: int, k: int) -> np.ndarray:
    x = np.asarray(v, np.float64).reshape(nreg, rb)[:, :span]
    return x.reshape(nreg, span // k, k).sum(-1).reshape(-1)


@contextlib.contextmanager
def haar_depth(J):
    """Inside: `score.py` computes the Haar-based metrics (`shared_specific`, `noise_floor`) at
    depth J, and the error budget (fixed to 2^14-bin blocks, not reported here) is NaN."""
    if J is None:
        yield
        return
    from onion_eval import error_budget as EB
    proxy = types.ModuleType("metrics_at_haar_depth")
    proxy.__dict__.update(Mx.__dict__)
    proxy.shared_specific = functools.partial(Mx.shared_specific, J=J)
    proxy.noise_floor = functools.partial(Mx.noise_floor, J=J)
    saved = (S.Mx, EB.error_budget, EB.noise_share_from_halves)
    S.Mx = proxy
    EB.error_budget = lambda *a, **k: {key: float("nan") for key in S.BUDGET_KEYS}
    EB.noise_share_from_halves = lambda *a, **k: None
    try:
        yield
    finally:
        S.Mx, EB.error_budget, EB.noise_share_from_halves = saved


class CacheScorer(S.Scorer):
    """`score.Scorer` whose per-target vectors are pooled to `at` bp and whose model vectors come
    from the prediction cache (module docstring)."""

    def __init__(self, info: dict, at: int, bound: str, **kw):
        if bound not in BOUNDS:
            raise ValueError(f"bound {bound!r} not in {BOUNDS}")
        cfg = {k: v for k, v in info["config"].items() if not k.startswith("res_")}
        super().__init__(info["evalset"], main="model", model=_NoModel(), config=cfg,
                         run_tables_sha256=info["tables_sha256"], **kw)
        self.info, self.at, self.bound = info, int(at), bound
        self.cache = Path(info["run"]) / CACHE
        self.span, self.J = span_of(self.at, self.rb)
        self.k = self.at // C.RES
        self.sig, self.sig_reason = None, "res_score.py scores the count space only"
        nreg = len(self.regions)
        self.strata = {name: m.reshape(nreg, self.rb)[:, :self.span]
                       .reshape(nreg, self.span // self.k, self.k).all(-1).reshape(-1)
                       for name, m in self.strata.items()}

    def _main_from_cache(self, target: str) -> tuple[np.ndarray, np.ndarray]:
        z = np.load(self.cache / f"{target}.npz")
        cb = int(z["bin_bp"])
        g = self.at // cb
        nb = self.span * C.RES // self.at
        stats = {}
        for key in ("mu_sum", "v_ind", "s_com"):
            a = np.asarray(z[key], np.float64)[:, : nb * g]
            stats[key] = a.reshape(a.shape[0], nb, g).sum(-1).reshape(-1)
        mu = stats["mu_sum"]
        n = mu * mu / (stats["v_ind"] if self.bound == "independent" else stats["s_com"] ** 2)
        return n, mu

    def _job_arrays(self, job, refs):
        A = super()._job_arrays(job, refs)
        nreg, rb, sp, k = len(self.regions), self.rb, self.span, self.k
        out = {}
        names = {key[:-3] for key in A if key.endswith(".mu")}
        for name in names:
            if name == "main":
                out["main.n"], out["main.mu"] = self._main_from_cache(job["target"])
                continue
            mu = torch.from_numpy(np.asarray(A[f"{name}.mu"], np.float64).reshape(nreg, rb)[:, :sp])
            n = torch.from_numpy(np.asarray(A[f"{name}.n"], np.float64).reshape(nreg, rb)[:, :sp])
            mp, np_ = pool_nb(mu, n, k, dim=1)
            out[f"{name}.mu"], out[f"{name}.n"] = mp.reshape(-1).numpy(), np_.reshape(-1).numpy()
        for key, v in A.items():
            if key in out or key.endswith((".mu", ".n")):
                continue
            if key in ("y", "x", "pr1", "pr2"):
                out[key] = _block_sum(v, nreg, rb, sp, k)
            elif key == "log2r":
                out[key] = v
            else:
                raise KeyError(f"res_score: no pooling rule for the vector {key!r}")
        return out

    def table(self) -> dict:
        imp, _, _, _ = self.jobs()
        imp = [j for j in imp if j.get("regime") == REGIME]
        missing = [j["target"] for j in imp if not (self.cache / f"{j['target']}.npz").is_file()]
        if missing:
            raise FileNotFoundError(f"{self.cache}: no prediction for {missing[:3]} "
                                    f"({len(missing)} targets); run predict again")
        sc = self.ev.get("scorable", {}).get("imputation", [])
        with haar_depth(self.J):
            self.marg = self._fit_marginal({j["assay"] for j in imp})
            groups = defaultdict(list)
            for j in imp:
                groups[j["assay"]].append(j)
            recs = []
            for a in sorted(groups):
                recs += self._imputation_group(groups[a], REGIME)
            m = self._imputation_metrics(recs, sc, require_all=True)
            unc, unc_den = self._uncertainty(recs, sc)
        if self.no_halves:
            imp_t = Mx.Undefined("--no-halves mode: headline 1 has no noise floor")
        else:
            imp_t = m["specific_variance_recovered_above_noise"][S.SPACE]["macro"]
        h1 = m["specific_variance_recovered_above_noise"][S.SPACE]
        out = {"imputation_gap_closed": S._j(imp_t),
               "imputation_reason": h1.get("reason", "") if S._undef(imp_t) else "",
               "imputation_per_assay": h1.get("per_assay", {}),
               "uncertainty_gap_closed": S._j(unc),
               "uncertainty_reason": S._why(unc) if S._undef(unc) else "",
               "uncertainty_marginal_pit_deviation": S._j(unc_den),
               "n_targets": len(recs)}
        for key in TABLE_KEYS:
            out[key] = m[key]["macro"]
            out[key + "_per_assay"] = m[key]["per_assay"]
            if m[key].get("reason"):
                out[key + "_reason"] = m[key]["reason"]
        return out


def table(run, at: int, out=None, *, bounds=BOUNDS, n_boot: int = 200, device: str = "cpu",
          log=print) -> dict:
    """The score table of a run at `at` bp from its prediction cache (module docstring)."""
    cache = Path(run) / CACHE
    if not (cache / INFO).is_file():
        raise FileNotFoundError(f"{cache / INFO}: no prediction cache; run `res_score.py "
                                f"predict` first (table never predicts)")
    info = json.loads((cache / INFO).read_text(encoding="utf-8"))
    at = int(at)
    if at not in SIZES:
        raise ValueError(f"--at {at} not in {SIZES}")
    ob, cb = int(info["output_bin_bp"]), int(info["cache_bin_bp"])
    if at % ob or at % cb:
        raise S.ScoreRefused(f"the arm's output bin is {ob} bp (cached at {cb} bp), which does "
                             f"not divide {at} bp: no {at} bp table for this arm")
    if _sha256(info["evalset"]) != info["evalset_sha256"]:
        raise S.ScoreRefused(f"{info['evalset']} changed since predict (sha256)")
    span, J = span_of(at, int(info["region_bins"]))
    need = span * C.RES // cb
    if int(info["n_cache_bins"]) < need:
        raise S.ScoreRefused(f"the cache covers {info['n_cache_bins']} bins of {cb} bp per "
                             f"region; the {at} bp table needs {need}")
    res = {"run": info["run"], "at_bp": at, "output_bin_bp": ob, "cache_bin_bp": cb,
           "span_bp_per_region": span * C.RES, "haar_depth_J": J if J is not None else Mx.J,
           "n_regions": len(info["regions"]), "evalset": info["evalset"],
           "evalset_sha256": info["evalset_sha256"], "tables_sha256": info["tables_sha256"],
           "regime": REGIME, "space": S.SPACE,
           "reference_predictors": "marginal_nb and position_average fitted at 25 bp, pooled by "
                                   "losses.pool_nb (independent size)",
           "bounds": {}}
    for bound in bounds:
        sc = CacheScorer(info, at, bound, n_boot=n_boot, device=device, log=log)
        res["bounds"][bound] = sc.table()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(res, indent=1, sort_keys=True, allow_nan=False) + "\n",
                       encoding="utf-8")
        tmp.replace(out)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--run", required=True)
    p.add_argument("--tables", required=True)
    p.add_argument("--evalset", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--infer-batch", type=int, default=8)
    p.add_argument("--force", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("--run", required=True)
    t.add_argument("--at", type=int, required=True, choices=SIZES)
    t.add_argument("--out", required=True)
    t.add_argument("--n-boot", type=int, default=200)
    t.add_argument("--bound", choices=("both",) + BOUNDS, default="both")
    t.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.cmd == "predict":
        print(predict(a.run, a.tables, a.evalset, device=a.device, infer_batch=a.infer_batch,
                      force=a.force))
        return 0
    bounds = BOUNDS if a.bound == "both" else (a.bound,)
    try:
        r = table(a.run, a.at, a.out, bounds=bounds, n_boot=a.n_boot, device=a.device)
    except S.ScoreRefused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps({b: {k: v for k, v in r["bounds"][b].items()
                          if k in ("imputation_gap_closed", "uncertainty_gap_closed",
                                   "pit_deviation_pooled", "count_crps")}
                      for b in r["bounds"]}))
    return 0


__all__ = ["BOUNDS", "CACHE", "CacheScorer", "SIZES", "haar_depth", "output_bin_bp", "predict",
           "span_of", "table"]

if __name__ == "__main__":
    sys.exit(main())
