"""t392 investigation: does smoothing the read-start model's start-space means recover its skill?

    python smooth_skill.py --run <run dir> --evalset <json> --assays H3K36me3 ... --out <dir>
        [--regions pair|all] [--max-items N] [--pred-store <seed store, for the W=1 check>]

Builds the scorer as `predict_store.py run` does (main=model, infer_batch 8, deviance on) and
replaces the `ReadStartAdapter.forward` of the loaded run by a copy that also smooths the inner
model's `mu_sub` along each stitched run (a centered moving mean over W bins per strand, the
window shrinking at the run ends; `n_sub` unchanged) before `read_start.convert_nb`. The W = 1
output is the adapter's own output (the same code), so the scorer's per-target records at W = 1
are the scorer's. Each (regime, assay) group is scored by `Scorer._imputation_group` once live
(W = 1; the W = 9 and 41 outputs are cached in call order) and once per W by replay.

Also collected at W = 1, per target: the start-space residuals y - mu_sub of the target track
(y = the target's read-start file named by the adapter's own kernel rule: `target_file`, `pr2`
for a halved track), their autocorrelation at lags 0..400 within regions, the inflation
c = 1 + offdiag / (diag - Poisson diag) of `conv_sparsity`, and E_nb of the converted NB with
n -> n / c.

Writes <out>/records.jsonl (one row per target and W) and <out>/resid.jsonl (one per target).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

WS = (1, 9, 41)
MAXLAG = 400


def smooth_run(ms: torch.Tensor, W: int) -> torch.Tensor:
    """[N, A, 2] -> centered moving mean over W rows along dim 0 (shrinking at the ends)."""
    if W <= 1:
        return ms
    N = ms.shape[0]
    c = torch.cat([torch.zeros_like(ms[:1]), ms.cumsum(0)], 0)
    i = torch.arange(N, device=ms.device)
    lo = (i - W // 2).clamp(0, N)
    hi = (i + W // 2 + 1).clamp(0, N)
    return (c[hi] - c[lo]) / (hi - lo).to(ms.dtype)[:, None, None]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--assays", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regions", choices=("pair", "all"), default="all")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--pred-store", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--from-store", action="store_true",
                    help="W = 1 only, from the stored predictions of --pred-store (no forward)")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    from onion_eval import score as S
    from onion_eval import deviance as DV
    from minicandi import read_start as RS
    from minicandi import contract as C
    from minicandi.registry import load_run
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import conv_sparsity as CS

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S.tf32_off()
    model, config = load_run(a.run)
    sc = S.Scorer(a.evalset, main="model", model=model, config=config, device="cpu",
                  run_tables_sha256=S.run_tables_sha256(a.run), infer_batch=8, deviance=True,
                  log=lambda *m: print(*m, flush=True))
    ev = json.loads(Path(a.evalset).read_text())
    if a.regions == "pair":
        idx = list(ev["pair_regions"])
        sc.regions = [sc.regions[i] for i in idx]
        sc.strata = {}                     # the strata masks cover all regions; unused here
    else:
        idx = list(range(len(sc.regions)))
    print(f"tile_bs {sc.tile_bs}, L {sc.L}, regions {len(sc.regions)}", flush=True)

    adapter = model                         # registry.load_run returns the ReadStartAdapter
    state = {"mode": "live", "cache": {W: [] for W in WS if W > 1}, "pos": 0, "t": None,
             "resid": [], "w1": []}

    def forward(self, batch):
        dev = batch["x_bin_mask"].device
        x_sub = self._inputs(batch).to(dev)
        b = dict(batch, x_counts_sub=x_sub, x_counts=x_sub.sum(-1, dtype=torch.int32))
        B, L, A = batch["x_bin_mask"].shape
        if self.config.get("average_input"):
            av = np.stack([self.ds.average_strands(m["cell"], m["chrom"], int(m["start_bin"]), L)
                           for m in batch["meta"]])
            avg_sub = torch.from_numpy(av).to(dev)
            b["avg_sub"], b["avg"] = avg_sub, avg_sub.sum(-1)
        o = self.inner(b)
        mu_sub, n_sub = o["mu_sub"], o["n_sub"]
        outs = {W: (torch.empty(B, L, A, dtype=mu_sub.dtype), torch.empty(B, L, A, dtype=mu_sub.dtype))
                for W in WS}
        t = state["t"]
        for run in RS.stitch(batch["meta"], L):
            ks = self._window_kernels(batch["meta"][run[0]])
            idx_ = torch.as_tensor(run)
            ms = mu_sub[idx_].reshape(len(run) * L, A, RS.STRANDS)
            ns = n_sub[idx_].reshape(len(run) * L, A, RS.STRANDS)
            for W in WS:
                msw = smooth_run(ms, W)
                for tt in range(A):
                    mt, nt = RS.convert_nb(msw[:, tt], ns[:, tt], ks[tt],
                                           stranded=not bool(self._un[tt]))
                    outs[W][0][idx_, :, tt] = mt.reshape(len(run), L)
                    outs[W][1][idx_, :, tt] = nt.reshape(len(run), L)
            # start-space residuals of the target track, per window of the run
            meta0 = batch["meta"][run[0]]
            halved = frozenset(meta0.get("halves_tracks") or ())
            mid = meta0["measurements_out"][t]
            which = ("pr2" if meta0["measurements_in"][t] and C.TRACKS[t] in halved
                     else meta0.get("target_file", "counts"))
            ys = np.concatenate([self.ds.read_strands(mid, batch["meta"][i]["chrom"],
                                                      int(batch["meta"][i]["start_bin"]), L, which)
                                 for i in run]).astype(np.float64)
            if bool(self._un[t]):
                ys = np.stack([ys.sum(1), np.zeros(len(ys))], 1)
            k = ks[t].numpy() if torch.is_tensor(ks[t]) else np.asarray(ks[t])
            k2 = k if k.ndim == 2 else np.stack([k, np.zeros_like(k)])
            state["resid"].append({"chrom": meta0["chrom"], "start": int(meta0["start_bin"]),
                                   "y": ys, "mu": ms[:, t].double().numpy(),
                                   "n": ns[:, t].double().numpy().clip(min=1e-3), "k": k2, "which": which,
                                   "mid": mid})
        self.n_windows += B
        for W in WS:
            if W > 1:
                state["cache"][W].append((outs[W][0][:, :, t].double().numpy(),
                                          outs[W][1][:, :, t].double().numpy(), A))
        state["w1"].append(outs[1][0][:, :, t].double().numpy().reshape(-1))
        return {"n": outs[1][1], "mu": outs[1][0]}

    adapter.forward = types.MethodType(forward, adapter)
    live_pred = sc.model_pred

    def replay(batch):
        W = state["mode"]
        m_t, n_t, A = state["cache"][W][state["pos"]]
        state["pos"] += 1
        mu = np.ones(m_t.shape + (A,)); n = np.ones(m_t.shape + (A,))   # only track t is read
        mu[:, :, state["t"]] = m_t
        n[:, :, state["t"]] = n_t
        return {"mu": mu, "n": n}

    sc._open_inputs()
    imp, cor, rep, ctl = sc.jobs()
    jobs = [j for j in imp if j["regime"] == "leave_one_assay_out" and j["assay"] in a.assays]
    if a.max_items:
        jobs = jobs[: a.max_items]
    sc.marg = sc._fit_marginal({j["assay"] for j in jobs})
    fr = open(out / "records.jsonl", "a")
    fz = open(out / "resid.jsonl", "a")
    for assay in a.assays:
        gj = [j for j in jobs if j["assay"] == assay]
        if not gj:
            continue
        t = C.TRACK_INDEX[assay]
        for j in gj:                       # one target at a time, so the caches stay small
            t0 = time.time()
            if a.from_store:
                key = f"item/{j['target']}/leave_one_assay_out"
                z = np.load(Path(a.pred_store) / (hashlib.sha1(key.encode()).hexdigest() + ".npz"))
                flat = {"mu": z["mu"][idx].astype(np.float64).ravel(), "n": z["n"][idx].astype(np.float64).ravel()}
                pos = {"i": 0}

                def from_store(batch, flat=flat, pos=pos, t=t):
                    B, L, A = batch["x_bin_mask"].shape
                    k = B * L
                    mu = np.ones((B, L, A)); n = np.ones((B, L, A))
                    mu[:, :, t] = flat["mu"][pos["i"]:pos["i"] + k].reshape(B, L)
                    n[:, :, t] = flat["n"][pos["i"]:pos["i"] + k].reshape(B, L)
                    pos["i"] += k
                    return {"mu": mu, "n": n}
                sc.model_pred = from_store
                r = sc._imputation_group([j], "leave_one_assay_out")[0]
                assert pos["i"] == flat["mu"].size, (pos["i"], flat["mu"].size)
                fr.write(json.dumps({"W": 1, "source": "store", "target": j["target"], "assay": assay,
                                     "regions": a.regions, "dev": r.get("dev")}) + "\n")
                fr.flush()
                print(f"{j['target']} {assay}: from store {time.time() - t0:.0f} s", flush=True)
                continue
            state.update(mode="live", t=t, resid=[], w1=[], cache={W: [] for W in WS if W > 1})
            sc.model_pred = live_pred
            recs = {1: sc._imputation_group([j], "leave_one_assay_out")[0]}
            t1 = time.time()
            for W in WS[1:]:
                state.update(mode=W, pos=0)
                sc.model_pred = replay
                recs[W] = sc._imputation_group([j], "leave_one_assay_out")[0]
            t2 = time.time()
            for W, r in recs.items():
                fr.write(json.dumps({"W": W, "target": j["target"], "assay": assay,
                                     "regions": a.regions, "dev": r.get("dev")}) + "\n")
            fr.flush()
            # W = 1 against the stored prediction of the scoring run
            chk = None
            if a.pred_store:
                key = f"item/{j['target']}/leave_one_assay_out"
                z = np.load(Path(a.pred_store) / (hashlib.sha1(key.encode()).hexdigest() + ".npz"))
                st = z["mu"][idx].astype(np.float64).ravel()
                mine = np.concatenate(state["w1"])
                chk = {"max_abs_diff_mu": float(np.abs(st - mine.astype(np.float32).astype(np.float64)).max()),
                       "n": int(st.size), "equal_float32": bool(np.array_equal(st, mine.astype(np.float32).astype(np.float64)))}
            # residual analysis: concatenate the runs of each region in genome order
            by = {}
            for rr in state["resid"]:
                by.setdefault(rr["chrom"], []).append(rr)
            num = np.zeros(MAXLAG + 1)
            den = 0.0
            sums = dict(convsq=0.0, diag=0.0, pois=0.0, od_model=0.0, sum_y=0.0, sum_mu=0.0,
                        sum_res2=0.0, sum_mu2n=0.0)
            numx = np.zeros(MAXLAG + 1)
            stranded = assay not in RS.UNSTRANDED_TRACKS
            ch = [0, 1] if stranded else [0]
            K = state["resid"][0]["k"]
            Dg = None
            for chrom, lst in by.items():
                lst.sort(key=lambda r: r["start"])
                # split into contiguous blocks
                blocks, cur = [], [lst[0]]
                for r in lst[1:]:
                    if r["start"] == cur[-1]["start"] + len(cur[-1]["y"]):
                        cur.append(r)
                    else:
                        blocks.append(cur); cur = [r]
                blocks.append(cur)
                for bl in blocks:
                    y = np.concatenate([r["y"] for r in bl]); mu = np.concatenate([r["mu"] for r in bl])
                    nn = np.concatenate([r["n"] for r in bl])
                    R = y - mu
                    R[:, [c for c in (0, 1) if c not in ch]] = 0.0
                    mu2 = mu.copy(); mu2[:, [c for c in (0, 1) if c not in ch]] = 0.0
                    od = mu2 ** 2 / nn                 # the model's own overdispersion per bin
                    od[:, [c for c in (0, 1) if c not in ch]] = 0.0
                    for c in ch:
                        x = R[:, c]
                        den += float(x @ x)
                        sums["sum_y"] += float(y[:, c].sum()); sums["sum_mu"] += float(mu[:, c].sum())
                        sums["sum_res2"] += float(x @ x); sums["sum_mu2n"] += float(od[:, c].sum())
                        for lag in range(1, MAXLAG + 1):
                            num[lag] += float(x[:-lag] @ x[lag:])
                    sl = slice(CS.REACH, len(R) - CS.REACH)
                    Q = CS.conv(R, K)
                    sums["convsq"] += float((Q[sl] ** 2).sum())
                    sums["diag"] += float(CS.conv(R ** 2, K ** 2)[sl].sum())
                    sums["pois"] += float(CS.conv(mu2, K ** 2)[sl].sum())
                    sums["od_model"] += float(CS.conv(od, K ** 2)[sl].sum())
            num[0] = den
            acf = num / den
            # lag covariance over the model's own claimed overdispersion (shot noise sits at lag 0)
            acf_od = num / sums["sum_mu2n"]
            acf_od[0] = 1.0
            c_inf = 1 + (sums["convsq"] - sums["diag"]) / (sums["diag"] - sums["pois"])
            c_model = 1 + (sums["convsq"] - sums["diag"]) / sums["od_model"]
            below = lambda thr: int(np.argmax(acf_od < thr)) if (acf_od < thr).any() else None
            fz.write(json.dumps({"target": j["target"], "assay": assay, "regions": a.regions,
                                 "which": state["resid"][0]["which"], "acf": acf.tolist(), "acf_od": acf_od.tolist(),
                                 "c_model": c_model,
                                 "lag_below_0.5": below(0.5), "lag_below_0.1": below(0.1),
                                 "c": c_inf, **sums, "w1_check": chk,
                                 "live_s": t1 - t0, "replay_s": t2 - t1}) + "\n")
            fz.flush()
            print(f"{j['target']} {assay}: live {t1 - t0:.0f} s, replay {t2 - t1:.0f} s, c {c_inf:.3f}, c_model {c_model:.3f}, "
                  f"lags {below(0.5)}/{below(0.1)}, check {chk}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
