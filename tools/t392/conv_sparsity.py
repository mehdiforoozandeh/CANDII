"""t392 investigation: two candidate causes of the read-start model's loss, on observed data.

    python conv_sparsity.py measure --ids <id> [<id> ...] --out <jsonl> [--kit DIR] [--nb-table F]
    python conv_sparsity.py summarize --jsonl <a.jsonl> [...] --out-dir DIR

(a) Conversion variance. `read_start.convert_nb` treats the start counts of neighbouring bins as
independent. Two designs, both on the pair regions of evalset_p16.json:
  half-difference: D = pr1 - pr2 per bin (starts per model channel, and overlap counts). The rate
    cancels, so this measures shot noise only.
  smooth predictor W: mu = box-smoothed pr1 starts (W bins), depth-scaled to pr2, as a stand-in
    for a model's prediction of pr2 (its errors are spatially correlated, as a model's are). An NB
    size n is fitted by moments; convert_nb (numpy port) gives the overlap NB; the deviance
    calibration ratio is computed with the scorer's own deviance.py, and compared with the same
    predictor built directly on overlap counts (as the read-count model predicts).
(b) Sparsity and replicate agreement: zero fraction, mean, Pearson pr1 vs pr2 (raw and log1p) of
starts, overlap counts, kernel-convolved starts and box-summed starts.

numpy, scipy, h5py only. Reads the read-start store and the eic_v1_2 overlap files; writes only --out.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

STORE = "/scratch/mforooz/rs392/store"
OVERLAP = "/scratch/mforooz/onion/measurements"
EVALSET = "/project/def-maxwl/mforooz/ONION/eval/eic_v1_2/evalset_p16.json"
TABLE = "/project/def-maxwl/mforooz/ONION/tables/eic_v1_2/measurements.tsv"
KIT = "/project/def-maxwl/mforooz/onion_kit/RS8"
UNSTRANDED = ("ATAC-seq", "DNase-seq")
REACH = 8
N_MIN, N_MAX = 1e-3, 1e4          # tools/t118/ladder/model.py, used by convert_nb's clamp
PAD = 256
WINDOWS = (9, 41, 401)


# ------------------------------------------------------------------ kernels and convolution
def model_channels(st: np.ndarray, stranded: bool) -> np.ndarray:
    """[m, 2] as the adapter folds it: an unstranded track holds the total in channel 0."""
    x = st.astype(np.float64)
    if not stranded:
        x = np.stack([x.sum(1), np.zeros(len(x))], 1)
    return x


def model_kernel(kernel, nr, stranded: bool) -> np.ndarray:
    """[2, 17]: the store kernel, or for an unstranded track the read-weighted mix in row 0."""
    k = np.asarray(kernel, np.float64)
    if stranded:
        return k
    nr = np.asarray(nr, np.float64)
    out = np.zeros_like(k)
    out[0] = (nr[:, None] * k).sum(0) / nr.sum()
    return out


def conv(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """a[i] = sum_{s,k} w[s, REACH + k] x[i - k, s]; zero outside the array (read_starts.conversion_floor)."""
    m = len(x)
    a = np.zeros(m)
    for s in range(2):
        for k in range(-REACH, REACH + 1):
            wk = w[s, REACH + k]
            if wk == 0:
                continue
            if k >= 0:
                a[k:] += wk * x[: m - k, s]
            else:
                a[: m + k] += wk * x[-k:, s]
    return a


def convert_nb_np(mu_sub: np.ndarray, n_sub: np.ndarray, w: np.ndarray, stranded: bool):
    """numpy port of read_start.convert_nb for one array [L, 2] (edge fill, clamps as there)."""
    L = len(mu_sub)
    pos = np.arange(L)
    n_sub = np.maximum(n_sub, N_MIN)
    mu = np.zeros(L)
    v = np.zeros(L)
    for k in range(-REACH, REACH + 1):
        src = np.clip(pos - k, 0, L - 1)
        for s in (range(2) if stranded else range(1)):
            c = w[s, REACH + k] * mu_sub[src, s]
            mu += c
            v += c * c / n_sub[src, s]
    n = np.where(v > 0, mu * mu / np.where(v > 0, v, 1.0), N_MAX)
    return mu, np.clip(n, N_MIN, N_MAX)


def box(x: np.ndarray, W: int) -> np.ndarray:
    """Centered moving mean over W bins along axis 0 (shrinking window at the ends)."""
    if W <= 1:
        return x.astype(np.float64)
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x.astype(np.float64)], 0), 0)
    h = W // 2
    i = np.arange(len(x))
    lo = np.clip(i - h, 0, len(x))
    hi = np.clip(i + h + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo).reshape((-1,) + (1,) * (x.ndim - 1))


def fit_size(y: np.ndarray, mu: np.ndarray) -> float:
    """Method-of-moments NB size: 1/n = sum((y - mu)^2 - mu) / sum(mu^2), clamped."""
    ex = float(((y - mu) ** 2 - mu).sum())
    s2 = float((mu ** 2).sum())
    if ex <= 0 or s2 <= 0:
        return N_MAX
    return float(np.clip(s2 / ex, N_MIN, N_MAX))


def pearson(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------------------ one measurement
def assay_of(ids):
    out = {}
    with open(TABLE) as f:
        head = f.readline().rstrip("\n").split("\t")
        ia, ii = head.index("assay"), head.index("measurement_id")
        for line in f:
            r = line.rstrip("\n").split("\t")
            if r[ii] in ids:
                out[r[ii]] = r[ia]
    return out


def regions(which: str):
    ev = json.loads(Path(EVALSET).read_text())
    regs = ev["regions"]
    if which == "pair":
        regs = [regs[i] for i in ev["pair_regions"]]
    return [(str(c), int(s)) for c, s in regs], int(ev["region_bins"])


def measure(mid: str, assay: str, regs, rb: int, dev) -> dict:
    stranded = assay not in UNSTRANDED
    sd = Path(STORE) / mid
    od = Path(OVERLAP) / mid
    f = {k: h5py.File(sd / f"starts{k}.h5", "r") for k in ("", "_pr1", "_pr2")}
    o = {k: h5py.File(od / f"{k}.h5", "r") for k in ("pr1", "pr2")}
    K_full = model_kernel(f[""].attrs["kernel"], f[""].attrs["n_reads_strand"], stranded)
    K_pr2 = model_kernel(f["_pr2"].attrs["kernel"], f["_pr2"].attrs["n_reads_strand"], stranded)
    depth = float(np.sum(f["_pr2"].attrs["n_reads_strand"])) / float(np.sum(f["_pr1"].attrs["n_reads_strand"]))
    ind = (np.asarray(K_full) > 0).astype(np.float64)
    act = [0, 1] if stranded else [0]

    acc = {}

    def add(key, val):
        acc[key] = acc.get(key, 0.0) + float(val)

    # per-bin arrays for (b) and the deviance pass, inner bins only
    keep = {k: [] for k in ("s1", "s2", "ov1", "ov2", "kc1", "kc2", "bx1", "bx2")}
    devin = {W: {k: [] for k in ("y_st", "mu_st", "y_ov", "mu_ov", "mu_od")} for W in WINDOWS}
    for chrom, start in regs:
        n = f[""][chrom].shape[0]
        lo, hi = max(0, start - PAD), min(n, start + rb + PAD)
        a, b = start - lo, start - lo + min(rb, n - start)
        x1 = model_channels(f["_pr1"][chrom][lo:hi], stranded)
        x2 = model_channels(f["_pr2"][chrom][lo:hi], stranded)
        v1 = np.asarray(o["pr1"][chrom][lo:hi], np.float64)
        v2 = np.asarray(o["pr2"][chrom][lo:hi], np.float64)
        sl = slice(a, b)
        # half-difference design
        D, S, Dov = x1 - x2, x1 + x2, v1 - v2
        cD = conv(D, K_full)
        add("hd_emp", (Dov[sl] ** 2).sum())
        add("hd_diag", conv(D ** 2, K_full ** 2)[sl].sum())
        add("hd_thin", (conv(S, K_full) + conv(D ** 2 - S, K_full ** 2))[sl].sum())
        add("hd_convsq", (cD[sl] ** 2).sum())
        add("hd_n", b - a)
        # (b) arrays
        keep["s1"].append(x1[sl][:, act]); keep["s2"].append(x2[sl][:, act])
        keep["ov1"].append(v1[sl]); keep["ov2"].append(v2[sl])
        keep["kc1"].append(conv(x1, K_full)[sl]); keep["kc2"].append(conv(x2, K_full)[sl])
        keep["bx1"].append(conv(x1, ind)[sl]); keep["bx2"].append(conv(x2, ind)[sl])
        # smooth-predictor design
        for W in WINDOWS:
            mu_st = box(x1, W) * depth
            mu_st[:, [c for c in (0, 1) if c not in act]] = 0.0
            R = x2 - mu_st
            devin[W]["y_st"].append(x2[sl][:, act]); devin[W]["mu_st"].append(mu_st[sl][:, act])
            devin[W]["y_ov"].append(v2[sl])
            devin[W]["mu_od"].append(box(v1, W)[sl] * depth)
            devin[W]["_R"] = devin[W].get("_R", []) + [(conv(R, K_pr2)[sl], conv(R ** 2, K_pr2 ** 2)[sl],
                                                       (v2 - conv(x2, K_pr2))[sl], conv(mu_st, K_pr2 ** 2)[sl])]
            devin[W]["_mu_full"] = devin[W].get("_mu_full", []) + [(mu_st, sl)]
    for h in list(f.values()) + list(o.values()):
        h.close()

    out = {"measurement": mid, "assay": assay, "stranded": stranded, "n_regions": len(regs),
           "n_bins": int(acc["hd_n"]), "depth_ratio_pr2_pr1": depth,
           "kernel_sum": float(K_full.sum()), "kernel_sq_sum": float((K_full ** 2).sum())}
    out.update({k: v for k, v in acc.items() if k != "hd_n"})

    # (b)
    cat = {k: np.concatenate(v) for k, v in keep.items()}
    tot1 = cat["s1"].sum(1); tot2 = cat["s2"].sum(1)
    series = {"starts_channel": (cat["s1"].ravel(), cat["s2"].ravel()),
              "starts_strandsum": (tot1, tot2), "overlap": (cat["ov1"], cat["ov2"]),
              "kernel_conv_starts": (cat["kc1"], cat["kc2"]), "box_sum_starts": (cat["bx1"], cat["bx2"])}
    for name, (p, q) in series.items():
        out[f"b_{name}_zero_frac"] = float(((p == 0).mean() + (q == 0).mean()) / 2)
        out[f"b_{name}_mean"] = float((p.mean() + q.mean()) / 2)
        out[f"b_{name}_r"] = pearson(p, q)
        out[f"b_{name}_r_log1p"] = pearson(np.log1p(p), np.log1p(q))
    out["b_r_overlap_vs_kernel_conv_pr1"] = pearson(cat["ov1"], cat["kc1"])

    # smooth-predictor design: fit n, convert, deviance ratios
    for W in WINDOWS:
        d = devin[W]
        y_st = np.concatenate(d["y_st"]); mu_st_in = np.concatenate(d["mu_st"])
        n_st = fit_size(y_st, mu_st_in)
        mu_ov_l, n_ov_l = [], []
        for mu_full, sl in d["_mu_full"]:
            m_, n_ = convert_nb_np(mu_full, np.full_like(mu_full, n_st), K_pr2, stranded)
            mu_ov_l.append(m_[sl]); n_ov_l.append(n_[sl])
        mu_ov = np.concatenate(mu_ov_l); n_ov = np.concatenate(n_ov_l)
        y_ov = np.concatenate(d["y_ov"]); mu_od = np.concatenate(d["mu_od"])
        Q = np.concatenate([r[0] for r in d["_R"]]); Dg = np.concatenate([r[1] for r in d["_R"]])
        F = np.concatenate([r[2] for r in d["_R"]])
        Dp = np.concatenate([r[3] for r in d["_R"]])
        var_ind = mu_ov + mu_ov ** 2 / n_ov
        n_od = fit_size(y_ov, mu_od)
        n_ref = fit_size(y_ov, mu_ov)            # convert's mean, size refitted in overlap space
        p = f"w{W}_"
        out[p + "n_start"] = n_st
        out[p + "n_overlap_direct"] = n_od
        out[p + "n_overlap_refit"] = n_ref
        out[p + "pearson_start"] = float(((y_st - mu_st_in) ** 2).sum() / (mu_st_in + mu_st_in ** 2 / n_st).sum())
        out[p + "emp"] = float(((y_ov - mu_ov) ** 2).sum())
        out[p + "var_indep"] = float(var_ind.sum())
        out[p + "convsq"] = float((Q ** 2).sum())
        out[p + "diag"] = float(Dg.sum())
        out[p + "diag_pois"] = float(Dp.sum())     # the Poisson part of diag: sum w^2 mu
        out[p + "floor_ss"] = float((F ** 2).sum())
        out[p + "cross_QF"] = float(2 * (Q * F).sum())
        um = lambda y, m: dev.unit_deviance(y, m).sum()
        en = lambda n_, m: dev.nb_expected_deviance(n_, m).sum()
        ys, ms = y_st.ravel(), mu_st_in.ravel()
        out[p + "devratio_start"] = float(um(ys, ms) / en(np.full_like(ms, n_st), ms))
        out[p + "devratio_convert"] = float(um(y_ov, mu_ov) / en(n_ov, mu_ov))
        out[p + "devratio_convert_refit"] = float(um(y_ov, mu_ov) / en(np.full_like(mu_ov, n_ref), mu_ov))
        out[p + "devratio_direct"] = float(um(y_ov, mu_od) / en(np.full_like(mu_od, n_od), mu_od))
        out[p + "dev_convert"] = float(um(y_ov, mu_ov) / len(y_ov))
        out[p + "dev_direct"] = float(um(y_ov, mu_od) / len(y_ov))
    return out


def cmd_measure(a) -> int:
    sys.path.insert(0, str(Path(a.kit) / "tools" / "onion_eval"))
    import deviance as dev
    if a.nb_table and Path(a.nb_table).exists():
        T = np.load(a.nb_table)
        dev._nb_table = lambda: T
    else:
        t0 = time.time()
        T = dev._nb_table()
        print(f"nb table built in {time.time() - t0:.0f} s", flush=True)
        if a.nb_table:
            np.save(a.nb_table, T)
    regs, rb = regions(a.regions)
    assays = assay_of(set(a.ids))
    with open(a.out, "a") as fo:
        for mid in a.ids:
            t0 = time.time()
            r = measure(mid, assays[mid], regs, rb, dev)
            r["regions_set"] = a.regions
            r["wall_s"] = time.time() - t0
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            print(f"{mid} {assays[mid]} {r['wall_s']:.1f} s", flush=True)
    return 0


# ------------------------------------------------------------------ summary
def cmd_summarize(a) -> int:
    import pandas as pd
    rows = [json.loads(l) for p in a.jsonl for l in open(p) if l.strip()]
    df = pd.DataFrame(rows)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_measurement.tsv", sep="\t", index=False)
    # (a) half-difference, per measurement then mean per assay
    df["hd_ratio_brief_w2"] = df.hd_emp / df.hd_diag
    df["hd_ratio_code_thin"] = df.hd_emp / df.hd_thin
    df["hd_offdiag_over_thin"] = (df.hd_convsq - df.hd_diag) / df.hd_thin
    for W in WINDOWS:
        p = f"w{W}_"
        df[p + "pearson_ratio"] = df[p + "emp"] / df[p + "var_indep"]
        df[p + "offdiag_over_var"] = (df[p + "convsq"] - df[p + "diag"]) / df[p + "var_indep"]
        # inflation of the overdispersion term v = sum c^2/n by neighbour covariance: n -> n / c_W
        df[p + "c_extra"] = 1 + (df[p + "convsq"] - df[p + "diag"]) / (df[p + "diag"] - df[p + "diag_pois"])
        df[p + "conv_inflation"] = df[p + "devratio_convert"] / df[p + "devratio_direct"]
    g = df.groupby("assay")
    cols_a = ["hd_ratio_brief_w2", "hd_ratio_code_thin", "hd_offdiag_over_thin"]
    for W in WINDOWS:
        p = f"w{W}_"
        cols_a += [p + c for c in ("pearson_start", "pearson_ratio", "offdiag_over_var", "c_extra", "devratio_start",
                                   "devratio_convert", "devratio_convert_refit", "devratio_direct",
                                   "conv_inflation", "dev_convert", "dev_direct")]
    ta = g[cols_a].mean()
    ta.insert(0, "n_bins", g["n_bins"].sum())
    ta.insert(0, "n_meas", g.size())
    allrow = df[cols_a].mean().to_frame().T
    allrow.insert(0, "n_bins", df["n_bins"].sum()); allrow.insert(0, "n_meas", len(df))
    allrow.index = ["ALL (mean over measurements)"]
    pd.concat([ta, allrow]).to_csv(out / "table_a.tsv", sep="\t", float_format="%.4f")
    cols_b = [c for c in df.columns if c.startswith("b_")]
    for c in ("starts_channel", "starts_strandsum"):
        df[f"r_drop_{c}"] = df["b_overlap_r"] - df[f"b_{c}_r"]
        df[f"r_log1p_drop_{c}"] = df["b_overlap_r_log1p"] - df[f"b_{c}_r_log1p"]
    cols_b += [c for c in df.columns if c.startswith("r_drop") or c.startswith("r_log1p_drop")]
    g = df.groupby("assay")
    tb = g[cols_b].mean()
    tb.insert(0, "n_bins", g["n_bins"].sum())
    tb.insert(0, "n_meas", g.size())
    allrow = df[cols_b].mean().to_frame().T
    allrow.insert(0, "n_bins", df["n_bins"].sum()); allrow.insert(0, "n_meas", len(df))
    allrow.index = ["ALL (mean over measurements)"]
    pd.concat([tb.sort_values("r_drop_starts_channel", ascending=False), allrow]).to_csv(
        out / "table_b.tsv", sep="\t", float_format="%.4f")
    print(f"{len(df)} measurements -> {out}")
    return 0


# ------------------------------------------------------------------ the model's own predictions
C_GRID = (1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)


def cmd_model(a) -> int:
    """Per leave-one-assay-out target: D0, D_model, and E_nb of the stored converted NB with the
    overdispersion term scaled by c (n -> n / c), for c on C_GRID, at the full-correlation bound
    c_full = (sum w)^2 / sum w^2, and at --factors (per assay c, from `summarize`)."""
    import hashlib
    import pandas as pd
    sys.path.insert(0, str(Path(a.kit) / "tools" / "onion_eval"))
    import deviance as dev
    T = np.load(a.nb_table)
    dev._nb_table = lambda: T
    ev = json.loads(Path(EVALSET).read_text())
    regs = [(str(c), int(s)) for c, s in ev["regions"]]
    rb = int(ev["region_bins"])
    tab = pd.read_csv(TABLE, sep="\t", dtype=str).set_index("measurement_id")
    fac = {}
    if a.factors:
        ft = pd.read_csv(a.factors, sep="\t", index_col=0)
        fac = {col: ft[col].to_dict() for col in ft.columns if col.endswith("c_extra")}
    jobs = [j for j in ev["imputation_targets"] if j["regime"] == "leave_one_assay_out"]
    with open(a.out, "a") as fo:
        for j in jobs:
            t0 = time.time()
            mid, assay = j["target"], j["assay"]
            key = f"item/{mid}/leave_one_assay_out"
            z = np.load(Path(a.pred_store) / (hashlib.sha1(key.encode()).hexdigest() + ".npz"))
            assert str(z["key"]) == key, (str(z["key"]), key)
            mu = z["mu"].astype(np.float64).ravel(); n = z["n"].astype(np.float64).ravel()
            cp = tab.loc[mid, "counts_path"]
            cp = cp[3:] if cp.startswith("h5:") else cp
            with h5py.File(cp, "r") as f:
                y = np.concatenate([np.asarray(f[c][s:s + rb], np.float64) for c, s in regs])
            stranded = assay not in UNSTRANDED
            with h5py.File(Path(STORE) / mid / "starts.h5", "r") as f:
                K = model_kernel(f.attrs["kernel"], f.attrs["n_reads_strand"], stranded)
            c_full = float(K.sum() ** 2 / (K ** 2).sum())
            mm = np.maximum(mu, dev.MU_MIN)
            r = {"target": mid, "assay": assay, "seed_store": a.pred_store, "bins": int(y.size),
                 "D0": dev.deviance(y, np.full_like(y, max(y.mean(), dev.MU_MIN))),
                 "D_model": dev.deviance(y, mu), "c_full": c_full,
                 "pearson_model": float(((y - mm) ** 2).sum() / (mm + mm ** 2 / n).sum())}
            for c in C_GRID:
                r[f"E_nb_c{c:g}"] = float(dev.nb_expected_deviance(n / c, mu).sum())
            r["E_nb_cfull"] = float(dev.nb_expected_deviance(n / c_full, mu).sum())
            for col, d in fac.items():
                c = float(d.get(assay, np.nan))
                if np.isfinite(c):
                    r[f"E_nb_{col}"] = float(dev.nb_expected_deviance(n / max(c, 1e-9), mu).sum())
                    r[f"{col}"] = c
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            print(f"{key} {time.time() - t0:.1f} s", flush=True)
    return 0


SCORABLE = ("H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me2",
            "H3K9ac", "H3K9me3")      # per_assay keys of dev_calibration_ratio in the p16_dev scores.json


def _E_at(r: dict, c: float) -> float:
    """E_nb of one record at overdispersion scale c: linear in log c between C_GRID points."""
    g = np.log(np.array(C_GRID))
    e = np.array([r[f"E_nb_c{x:g}"] for x in C_GRID])
    return float(np.interp(np.log(max(c, 1.0)), g, e))


def _macro(recs, cfun) -> tuple:
    """(macro over SCORABLE, per-assay) of mean(D_model/D0) / mean(E_nb(c)/D0), as score.py."""
    per = {}
    for a in SCORABLE:
        rs = [r for r in recs if r["assay"] == a]
        if rs:
            per[a] = np.mean([r["D_model"] / r["D0"] for r in rs]) / np.mean(
                [cfun(r) / r["D0"] for r in rs])
    return float(np.mean(list(per.values()))), per


def cmd_link(a) -> int:
    """Calibration ratio of the stored predictions at c = 1 (must equal the scorer), at c_full and
    at the per-assay c_extra of table_a; and the common c that brings the macro to --target."""
    import pandas as pd
    ta = pd.read_csv(a.table_a, sep="\t", index_col=0)
    rows = []
    for path in a.jsonl:
        recs = [json.loads(l) for l in open(path) if l.strip()]
        row = {"file": Path(path).name, "n_targets": len(recs)}
        row["ratio_c1"] = _macro(recs, lambda r: r["E_nb_c1"])[0]
        row["ratio_cfull"] = _macro(recs, lambda r: r["E_nb_cfull"])[0]
        for W in WINDOWS:
            col = f"w{W}_c_extra"
            row[f"ratio_cextra_w{W}"] = _macro(recs, lambda r: _E_at(r, float(ta.loc[r["assay"], col])))[0]
        lo, hi = 1.0, C_GRID[-1]
        for _ in range(60):
            mid = (lo * hi) ** 0.5
            v = _macro(recs, lambda r: _E_at(r, mid))[0]
            lo, hi = (mid, hi) if v > a.target else (lo, mid)
        row[f"c_to_reach_{a.target:g}"] = (lo * hi) ** 0.5
        row["mean_c_full_scorable"] = float(np.mean([r["c_full"] for r in recs if r["assay"] in SCORABLE]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, sep="\t", index=False, float_format="%.4f")
    print(df.to_string())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("measure")
    m.add_argument("--ids", nargs="+", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--regions", choices=("pair", "all"), default="pair")
    m.add_argument("--kit", default=KIT)
    m.add_argument("--nb-table", default=None)
    md = sp.add_parser("model")
    md.add_argument("--pred-store", required=True)
    md.add_argument("--out", required=True)
    md.add_argument("--nb-table", required=True)
    md.add_argument("--factors", default=None)
    md.add_argument("--kit", default=KIT)
    lk = sp.add_parser("link")
    lk.add_argument("--jsonl", nargs="+", required=True)
    lk.add_argument("--table-a", required=True)
    lk.add_argument("--target", type=float, default=1.0584)
    lk.add_argument("--out", required=True)
    s = sp.add_parser("summarize")
    s.add_argument("--jsonl", nargs="+", required=True)
    s.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    return {"measure": cmd_measure, "model": cmd_model, "summarize": cmd_summarize, "link": cmd_link}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
