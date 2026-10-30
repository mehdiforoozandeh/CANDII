"""t136 readings (h28, the arm-to-arm training grid): the tables and the readings of checks c1-c5.

Inputs (in <dir>, brought down by run_all.sh; small tables only):
  records_nibi.tsv.gz, runs_nibi.tsv          the Darm1 / Darm2 grid (extract_records.py grid, Nibi)
  ref_records_fir.tsv.gz, ref_runs_fir.tsv    Dref (phase-2 old D direct, per-track g) and Dviabase
                                              (Dref's two maps through the base) (extract_records.py ref, Fir)
Outputs (in <dir>):
  n_table.tsv            per arm: runs assigned, run dirs, marks, finished, pair counts, best steps
  m5_depthlaw_table.tsv  median |log2 error| of the depth-law scale per scope x class x arm x group
  m1_heldout_table.tsv   pooled CRPS (with the oracle_scaled / scale_error split) on the law pairs,
                         per scope x class x arm x subgroup x knob
  m3_trained_table.tsv   pooled CRPS of the trained base<->arm pairs and of base<->7.5M
  checks_readings.tsv    the readings of h28's c1-c5 and of the combination rule (READINGS, no ticks)
  READINGS.md, DIGEST.md the same, for a human

Definitions (h28 "Verifiables", plan/T136_ARM2ARM_GRID.md section 5 and "Checks as written"):
  groups of the depth law (kind depthlaw): trained_base = base <-> 15M, base <-> 3.75M
    (depth_source trained); heldout_base = base <-> 7.5M; heldout_r2 = the 4 law pairs that touch
    7.5M (ratio 2; t128p2's never75); trained_r4 = 15M <-> 3.75M (ratio 4; trained in Darm1 and
    Darm2 = in config["extra_pairs"]; never-trained in Dref and Dviabase; column `status`).
  The depth-law value is the median of |log2_pred - log2_true| over every row of the cell (all
    seeds, all tracks), as t128p2's m5 (the reference values 0.195, 0.125, 0.052 are made this
    way). The per-seed medians give the seed spread; their mean is beside it (mean_seed_medians).
  Held-out arm->arm pairs (m1 subgroup heldout75) = every law pair with depth 7.5M as source or
    target, all knobs. knob = the arm of the other product (depth for the 4 depth pairs).
  Pooled CRPS (t118): per seed, mean over a track's pairs, then mean over tracks; the reading is
    the mean over seeds.
  Seed spread = max - min over seeds 0, 1, 2 of the per-seed value. Noise label of a gap:
    "counts" when |gap| > 2 x the larger seed spread, else "within noise"; "incomplete" when a
    side does not hold 3 seeds of every track of the scope.
  Scopes: all7 = the 7 tracks; darm2_tracks = the tracks where Darm2 has all 3 seeds in that
    group (a partial reading while Darm2 is unfinished). Classes: all, histone (6), dnase (C12M02).
Checks: the thresholds the PI accepted on 2026-10-29 (C1_MIN ... C5_MAX_LAST10 below; h28).

    python analyze.py [<dir>]       (default: this file's directory)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ["Dref", "Dviabase", "Darm1", "Darm2"]
GRID = ["Darm1", "Darm2"]
METRICS = ("crps_all", "crps_oracle_scaled_all", "scale_error_all")
PAIR = ["track", "arm_src", "level_src", "arm_tgt", "level_tgt"]
SEEDS = (0, 1, 2)
TRACKS = ("C07M20", "C07M29", "C12M02", "C19M16", "C19M22", "C40M17", "C40M18")
DNASE = "C12M02"
CLASSES = {"all": TRACKS, "histone": tuple(t for t in TRACKS if t != DNASE), "dnase": (DNASE,)}
CAP, LAST10 = 12000, 10800          # c5: best step > 10 800 of the 12 000-step cap
C1_MIN, C4_CLOSE, C4_OS, K_NOISE, C5_MAX_LAST10 = 0.5, 0.5, 0.25, 2.0, 2


def extra_expected(track, arm):
    return 2 if arm == "Darm1" else (30 if track == DNASE else 306)


def read(here, name):
    f = Path(here) / name
    return pd.read_csv(f, sep="\t", low_memory=False) if f.is_file() else pd.DataFrame()


def load(here):
    d = pd.concat([read(here, "records_nibi.tsv.gz"), read(here, "ref_records_fir.tsv.gz")],
                  ignore_index=True)
    for c in list(METRICS) + ["log2_true", "log2_pred", "seed", "is_extra"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["seed"] = d.seed.astype(int)
    d["is_extra"] = d.is_extra.fillna(0).astype(int)
    for c in ("level_src", "level_tgt"):
        d[c] = d[c].astype(str)
    assert d.space.eq("counts").all() and d.model.eq("real").all() and d.g.ne("all").all()
    d = d[d.arm.isin(ARMS)].copy()
    d["touch75"] = (((d.arm_src == "depth") & (d.level_src == "7.5M"))
                    | ((d.arm_tgt == "depth") & (d.level_tgt == "7.5M")))
    d["depthdepth"] = (d.arm_src == "depth") & (d.arm_tgt == "depth")
    runs = pd.concat([read(here, "runs_nibi.tsv"), read(here, "ref_runs_fir.tsv")],
                     ignore_index=True)
    for c in ("TRAIN_DONE", "SCORE_DONE", "LAW_DONE", "run_dir"):
        runs[c] = pd.to_numeric(runs[c], errors="coerce").fillna(0).astype(int)
    for c in ("seed", "max_steps", "steps_run", "best_step", "n_train_pairs", "n_train_pairs_75",
              "n_extra_pairs", "n_extra_pairs_75"):
        runs[c] = pd.to_numeric(runs[c], errors="coerce")
    return d, runs


# ------------------------------------------------------------------------------------- noise


def spread(s):
    s = pd.Series(s, dtype=float).dropna()
    return float(s.max() - s.min()) if len(s) >= 2 else np.nan


def noise_label(gap, sa, sb, ca, cb):
    if pd.isna(gap):
        return "n/a"
    if not (ca and cb) or pd.isna(sa) or pd.isna(sb):
        return "incomplete"
    return "counts" if abs(gap) > K_NOISE * max(sa, sb) else "within noise"


def add_gaps(t, keys, value, refs=("Dref", "Dviabase"), closed=()):
    """gap_vs_<ref> = arm - ref, spread_<ref>, label_vs_<ref>; and, for each column c in
    `closed`, closed_<c> = (Dref - arm) / (Dref - Dviabase) of that column."""
    t = t.copy()
    idx = {ref: t[t.arm == ref].set_index(keys) for ref in refs}
    for ref in refs:
        g, sb, lb = [], [], []
        for _, row in t.iterrows():
            k = tuple(row[c] for c in keys)
            r = idx[ref]
            if row.arm == ref or k not in r.index:
                g.append(np.nan); sb.append(np.nan); lb.append("n/a")
                continue
            rr = r.loc[k]
            gap = row[value] - rr[value]
            g.append(gap); sb.append(rr.seed_spread)
            lb.append(noise_label(gap, row.seed_spread, rr.seed_spread, row.complete, rr.complete))
        t[f"gap_vs_{ref}"], t[f"spread_{ref}"], t[f"label_vs_{ref}"] = g, sb, lb
    for c in closed:
        out = []
        for _, row in t.iterrows():
            k = tuple(row[c2] for c2 in keys)
            if k in idx["Dref"].index and k in idx["Dviabase"].index:
                a, b = idx["Dref"].loc[k][c], idx["Dviabase"].loc[k][c]
                out.append((a - row[c]) / (a - b) if a != b else np.nan)
            else:
                out.append(np.nan)
        t[f"closed_{c}"] = out
    return t


def scoped(x):
    """Yield (scope, class, rows): all7, and darm2_tracks = tracks where Darm2 holds 3 seeds of
    this selection (skipped when that set is empty or all 7)."""
    d2 = x[x.arm == "Darm2"].groupby("track").seed.nunique()
    d2 = tuple(sorted(d2[d2 == len(SEEDS)].index))
    for scope, tracks in (("all7", TRACKS), ("darm2_tracks", d2)):
        if scope == "darm2_tracks" and (not d2 or d2 == TRACKS):
            continue
        for cls, ct in CLASSES.items():
            keep = tuple(t for t in tracks if t in ct)
            if keep:
                yield scope, cls, keep, x[x.track.isin(keep)]


def complete_of(s, keep):
    have = s.groupby("track").seed.nunique()
    return bool(set(keep) <= set(have[have == len(SEEDS)].index))


# ------------------------------------------------------------------------------------- tables


def depth_rows(d):
    x = d[d.kind == "depthlaw"].copy()
    law = d[d.kind == "law"]
    extra = set(map(tuple, law[law.is_extra == 1][["run"] + PAIR].values.tolist()))
    x["is_extra"] = [int(tuple(r) in extra) for r in x[["run"] + PAIR].values.tolist()]
    r4 = x.depthdepth & ~x.touch75 & (x.depth_source == "law")
    x["group"] = np.select([x.depth_source == "trained", x.depth_source == "heldout",
                            (x.depth_source == "law") & x.touch75, r4],
                           ["trained_base", "heldout_base", "heldout_r2", "trained_r4"],
                           "never_other")
    bad = x[(x.group == "trained_r4") & x.arm.isin(GRID) & (x.is_extra == 0)]
    assert bad.empty, f"{len(bad)} grid 15M<->3.75M rows not in config extra_pairs"
    x["abs_err"] = (x.log2_pred - x.log2_true).abs()
    x["err"] = x.log2_pred - x.log2_true
    return x


def m5_depthlaw(d):
    x = depth_rows(d)
    out = []
    for grp, xg in x.groupby("group"):
        for scope, cls, keep, s0 in scoped(xg):
            for arm, s in s0.groupby("arm"):
                ps = s.groupby("seed").abs_err.median()
                rec = {"scope": scope, "class": cls, "arm": arm, "group": grp,
                       "status": ("trained" if (s.is_extra == 1).all() else "never-trained")
                       if grp == "trained_r4" else
                       {"trained_base": "trained", "heldout_base": "held out",
                        "heldout_r2": "held out"}.get(grp, "never-trained"),
                       "median_abs_err": float(s.abs_err.median()),
                       "seed_spread": spread(ps), "n_seeds": int(ps.notna().sum()),
                       "mean_seed_medians": float(ps.mean()),
                       "median_err": float(s.err.median()),
                       "n_runs": s.run.nunique(), "n_tracks": s.track.nunique(),
                       "n_rows": int(s.abs_err.notna().sum()),
                       "n_none": int(s.abs_err.isna().sum()),
                       "complete": complete_of(s, keep), "tracks": ",".join(keep)}
                for sd in SEEDS:
                    rec[f"median_abs_err_s{sd}"] = ps.get(sd, np.nan)
                out.append(rec)
    t = pd.DataFrame(out)
    t = add_gaps(t, ["scope", "class", "group"], "median_abs_err", closed=("median_abs_err",))
    return order(t, ["scope", "class", "group"])


def pooled(p, keys, keep):
    tr = p.groupby(keys + ["seed", "track"])[list(METRICS)].mean()
    ps = tr.groupby(keys + ["seed"])[list(METRICS)].mean().reset_index()
    ntr = p.groupby(keys + ["seed"]).track.nunique()
    out = []
    for k, s in ps.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        sub = p
        for c, v in zip(keys, k):
            sub = sub[sub[c] == v]
        rec = dict(zip(keys, k))
        for m in METRICS:
            rec[m] = float(s[m].mean())
        rec["seed_spread"] = spread(s.crps_all)
        rec["seed_spread_oracle_scaled"] = spread(s.crps_oracle_scaled_all)
        rec["n_seeds"] = len(s)
        for sd in SEEDS:
            v = s[s.seed == sd].crps_all
            rec[f"crps_s{sd}"] = float(v.iloc[0]) if len(v) else np.nan
        rec["n_pairs"] = len(sub.drop_duplicates(PAIR))
        rec["n_records"] = len(sub)
        rec["n_runs"] = sub.run.nunique()
        rec["n_tracks_per_seed"] = ",".join(str(int(ntr.loc[k + (sd,)])) if k + (sd,) in ntr.index
                                            else "0" for sd in SEEDS)
        rec["complete"] = complete_of(sub, keep)
        rec["tracks"] = ",".join(keep)
        out.append(rec)
    return out


def m1_heldout(d):
    law = d[d.kind == "law"].copy()
    h = law[law.touch75].copy()
    src75 = (h.arm_src == "depth") & (h.level_src == "7.5M")
    h["knob"] = np.where(src75, h.arm_tgt, h.arm_src)
    parts = [h.assign(subgroup="heldout75"), h.assign(subgroup="heldout75", knob="all"),
             h[h.depthdepth].assign(subgroup="heldout75_depth", knob="all"),
             h[~h.depthdepth].assign(subgroup="heldout75_cross", knob="all"),
             law[~law.touch75 & (law.is_extra == 1)].assign(subgroup="extra_trained", knob="all"),
             law[~law.touch75 & (law.is_extra == 0)].assign(subgroup="never_other", knob="all")]
    x = pd.concat(parts, ignore_index=True)
    out = []
    for (sg, kb), xs in x.groupby(["subgroup", "knob"]):
        for scope, cls, keep, s0 in scoped(xs):
            if kb != "all" and cls != "all":
                continue
            for rec in pooled(s0, ["arm"], keep):
                rec.update(scope=scope, **{"class": cls}, subgroup=sg, knob=kb)
                out.append(rec)
    t = pd.DataFrame(out)
    t = add_gaps(t, ["scope", "class", "subgroup", "knob"], "crps_all",
                 closed=METRICS)
    lead = ["scope", "class", "arm", "subgroup", "knob"]
    t = t[lead + [c for c in t.columns if c not in lead]]
    return order(t, ["scope", "class", "subgroup", "knob"])


def m3_trained(d):
    tr = d[(d.kind == "trained") & (d.direction != "arm_to_arm")].assign(group="trained_base")
    he = d[d.kind == "heldout"].assign(group="heldout_base")
    out = []
    for grp, xs in pd.concat([tr, he], ignore_index=True).groupby("group"):
        for scope, cls, keep, s0 in scoped(xs):
            for rec in pooled(s0, ["arm"], keep):
                rec.update(scope=scope, **{"class": cls}, group=grp)
                out.append(rec)
    t = pd.DataFrame(out)
    t = add_gaps(t, ["scope", "class", "group"], "crps_all", refs=("Dref",))
    lead = ["scope", "class", "arm", "group"]
    t = t[lead + [c for c in t.columns if c not in lead]]
    return order(t, ["scope", "class", "group"])


def n_table(runs):
    out = []
    for arm in ARMS:
        r = runs[runs.arm == arm]
        via = arm == "Dviabase"
        fin = r.LAW_DONE == 1 if via else (r.TRAIN_DONE + r.SCORE_DONE + r.LAW_DONE == 3)
        cfg = r[r.n_train_pairs.notna()]
        rec = {"arm": arm, "assigned": len(r), "run_dirs": int(r.run_dir.sum()),
               "TRAIN_DONE": int(r.TRAIN_DONE.sum()), "SCORE_DONE": int(r.SCORE_DONE.sum()),
               "LAW_DONE": int(r.LAW_DONE.sum()), "finished": int(fin.sum()),
               "n_configs": len(cfg)}
        if not via:
            tr = r[r.TRAIN_DONE == 1]
            rec.update(n_runs_train75=int((cfg.n_train_pairs_75 > 0).sum()),
                       n_runs_extra75=int((cfg.n_extra_pairs_75.fillna(0) > 0).sum()),
                       n_best_last10=int((tr.best_step > LAST10).sum()),
                       n_at_cap=int((tr.steps_run >= CAP).sum()),
                       best_step_median=tr.best_step.median(), best_step_min=tr.best_step.min(),
                       best_step_max=tr.best_step.max(), steps_run_median=tr.steps_run.median(),
                       max_steps=",".join(str(int(v)) for v in sorted(tr.max_steps.dropna().unique())))
            if arm in GRID:
                ok = [int(n) == extra_expected(g, arm) for g, n in zip(cfg.g, cfg.n_extra_pairs)]
                rec.update(n_extra_ok=int(sum(ok)),
                           extra_counts=";".join(f"{g}:{int(n)}" for g, n in
                                                 sorted(set(zip(cfg.g, cfg.n_extra_pairs)))))
            else:
                rec.update(n_extra_ok=int((cfg.n_extra_pairs == 0).sum()), extra_counts="0")
        out.append(rec)
    return pd.DataFrame(out)


def order(t, by):
    t = t.copy()
    t["_a"] = t.arm.map({a: i for i, a in enumerate(ARMS)})
    return t.sort_values(by + ["_a"]).drop(columns="_a").reset_index(drop=True)


# ------------------------------------------------------------------------------------- checks


def cell(t, **sel):
    s = t
    for c, v in sel.items():
        s = s[s[c] == v]
    return s.iloc[0] if len(s) == 1 else None


def f3(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:.3f}"


def both(rows):
    rs = [r for r in rows]
    if any(r == "FAIL" for r in rs):
        return "FAIL"
    if any(r == "NOT-READABLE" for r in rs):
        return "NOT-READABLE"
    return "PASS"


def checks(m5, m1, m3, nt):
    rows = []

    def add(check, name, kind, arm, item, value, threshold, reading, detail):
        rows.append({"check": check, "name": name, "kind": kind, "arm": arm, "item": item,
                     "value": value, "threshold": threshold, "reading": reading, "detail": detail})

    D5 = dict(scope="all7", **{"class": "all"})
    # c1 lawheldout
    ref, via, a = (cell(m5, arm=x, group="heldout_r2", **D5) for x in ("Dref", "Dviabase", "Darm1"))
    ok = all(r is not None and r.complete for r in (ref, via, a))
    v = (ref.median_abs_err - a.median_abs_err) / (ref.median_abs_err - via.median_abs_err) if ok else np.nan
    bar = ref.median_abs_err - C1_MIN * (ref.median_abs_err - via.median_abs_err) if ok else np.nan
    add("c1", "lawheldout", "claim", "Darm1", "gap closed on heldout_r2", v, f">= {C1_MIN}",
        ("PASS" if v >= C1_MIN else "FAIL") if ok else "NOT-READABLE",
        f"median |log2 err| Dref {f3(ref.median_abs_err)} (spread {f3(ref.seed_spread)}), "
        f"Dviabase {f3(via.median_abs_err)} (spread {f3(via.seed_spread)}), "
        f"Darm1 {f3(a.median_abs_err)} (spread {f3(a.seed_spread)}; per seed "
        f"{', '.join(f3(a[f'median_abs_err_s{sd}']) for sd in SEEDS)}); pass needs Darm1 <= {f3(bar)}; "
        f"no noise clause (design option A); on the mean of the per-seed medians the gap closed is "
        f"{f3((ref.mean_seed_medians - a.mean_seed_medians) / (ref.mean_seed_medians - via.mean_seed_medians))}"
        if ok else "a cell is incomplete")
    # c2 basefit
    c2 = {}
    for arm in GRID:
        parts = []
        for item, tab, sel in (("trained_base median |log2 err|", m5, dict(group="trained_base")),
                               ("trained base<->arm crps_all", m3, dict(group="trained_base"))):
            col = "median_abs_err" if tab is m5 else "crps_all"
            r_ref, r_a = cell(tab, arm="Dref", **D5, **sel), cell(tab, arm=arm, **D5, **sel)
            if r_a is not None and r_a.complete and r_ref is not None and r_ref.complete:
                gap = abs(r_a[col] - r_ref[col])
                thr = K_NOISE * max(r_a.seed_spread, r_ref.seed_spread)
                rd = "PASS" if gap <= thr else "FAIL"
                det = (f"{arm} {f3(r_a[col])} (spread {f3(r_a.seed_spread)}) vs Dref "
                       f"{f3(r_ref[col])} (spread {f3(r_ref.seed_spread)}); |gap| {f3(gap)} vs "
                       f"band {f3(thr)}")
                if tab is m3:
                    det += (f"; split {arm} oracle_scaled {f3(r_a.crps_oracle_scaled_all)}, "
                            f"scale_error {f3(r_a.scale_error_all)}; Dref "
                            f"{f3(r_ref.crps_oracle_scaled_all)}, {f3(r_ref.scale_error_all)}")
                add("c2", "basefit", "control", arm, item, gap, f"<= {f3(thr)}", rd, det)
            else:
                pr = cell(tab, arm=arm, scope="darm2_tracks", **{"class": "all"}, **sel)
                pf = cell(tab, arm="Dref", scope="darm2_tracks", **{"class": "all"}, **sel)
                det = f"{arm} incomplete in all7 ({0 if r_a is None else r_a.n_runs} of 21 runs scored)"
                if pr is not None and pf is not None:
                    g2 = abs(pr[col] - pf[col])
                    t2 = K_NOISE * max(pr.seed_spread, pf.seed_spread)
                    det += (f"; partial on tracks {pr.tracks}: {arm} {f3(pr[col])} (spread "
                            f"{f3(pr.seed_spread)}) vs Dref {f3(pf[col])} (spread "
                            f"{f3(pf.seed_spread)}), |gap| {f3(g2)} vs band {f3(t2)} "
                            f"-> {'within' if g2 <= t2 else 'outside'} the band")
                add("c2", "basefit", "control", arm, item, np.nan, "<= 2 x larger spread",
                    "NOT-READABLE", det)
            parts.append(rows[-1]["reading"])
        c2[arm] = both(parts)
        add("c2", "basefit", "control", arm, "arm reading", np.nan, "both items", c2[arm],
            "a FAIL invalidates this arm's run")
    add("c2", "basefit", "control", "both", "check reading", np.nan, "Darm1 and Darm2",
        both(c2.values()), f"Darm1 {c2['Darm1']}, Darm2 {c2['Darm2']}")
    # c3 ratiotransfer
    h, t = cell(m5, arm="Darm1", group="heldout_r2", **D5), cell(m5, arm="Darm1", group="trained_r4", **D5)
    ok = h is not None and t is not None and h.complete and t.complete
    thr = t.median_abs_err + K_NOISE * max(h.seed_spread, t.seed_spread) if ok else np.nan
    add("c3", "ratiotransfer", "claim", "Darm1", "heldout_r2 vs trained_r4 + 2 x spread",
        h.median_abs_err if ok else np.nan, f"<= {f3(thr)}",
        ("PASS" if h.median_abs_err <= thr else "FAIL") if ok else "NOT-READABLE",
        f"heldout_r2 {f3(h.median_abs_err)} (spread {f3(h.seed_spread)}); trained_r4 "
        f"{f3(t.median_abs_err)} (spread {f3(t.seed_spread)}); bar = {f3(t.median_abs_err)} + 2 x "
        f"{f3(max(h.seed_spread, t.seed_spread))} = {f3(thr)}" if ok else "a cell is incomplete")
    # c4 crpsheldout
    S1 = dict(scope="all7", **{"class": "all"}, subgroup="heldout75", knob="all")
    ref, via, a = (cell(m1, arm=x, **S1) for x in ("Dref", "Dviabase", "Darm1"))
    ok = all(r is not None and r.complete for r in (ref, via, a))
    if ok:
        cl = (ref.crps_all - a.crps_all) / (ref.crps_all - via.crps_all)
        gain, gthr = ref.crps_all - a.crps_all, K_NOISE * max(a.seed_spread, ref.seed_spread)
        osc = ((ref.crps_oracle_scaled_all - a.crps_oracle_scaled_all)
               / (ref.crps_oracle_scaled_all - via.crps_oracle_scaled_all))
        r1, r2, r3 = ("PASS" if cl >= C4_CLOSE else "FAIL", "PASS" if gain > gthr else "FAIL",
                      "PASS" if osc >= C4_OS else "FAIL")
        split = (f"crps_all Dref {f3(ref.crps_all)} (oracle_scaled {f3(ref.crps_oracle_scaled_all)}, "
                 f"scale_error {f3(ref.scale_error_all)}; spread {f3(ref.seed_spread)}), Dviabase "
                 f"{f3(via.crps_all)} ({f3(via.crps_oracle_scaled_all)}, {f3(via.scale_error_all)}; "
                 f"spread {f3(via.seed_spread)}), Darm1 {f3(a.crps_all)} "
                 f"({f3(a.crps_oracle_scaled_all)}, {f3(a.scale_error_all)}; spread "
                 f"{f3(a.seed_spread)}); {int(a.n_pairs)} pairs")
        add("c4", "crpsheldout", "claim", "Darm1", "crps_all gap closed", cl, f">= {C4_CLOSE}", r1, split)
        add("c4", "crpsheldout", "claim", "Darm1", "gain Dref - Darm1", gain, f"> {f3(gthr)}", r2,
            f"gain {f3(gain)} vs 2 x max(spread Darm1 {f3(a.seed_spread)}, Dref {f3(ref.seed_spread)})")
        add("c4", "crpsheldout", "claim", "Darm1", "oracle_scaled gap closed", osc, f">= {C4_OS}", r3,
            f"oracle_scaled Dref {f3(ref.crps_oracle_scaled_all)}, Dviabase "
            f"{f3(via.crps_oracle_scaled_all)}, Darm1 {f3(a.crps_oracle_scaled_all)}")
        add("c4", "crpsheldout", "claim", "Darm1", "check reading", np.nan, "all three",
            both([r1, r2, r3]), split)
    else:
        add("c4", "crpsheldout", "claim", "Darm1", "check reading", np.nan, "all three",
            "NOT-READABLE", "a cell is incomplete")
    # c5 runhealth
    c5 = {}
    dref = nt[nt.arm == "Dref"].iloc[0]
    for arm in GRID:
        r = nt[nt.arm == arm].iloc[0]
        fin = "PASS" if r.finished == 21 else "NOT-READABLE"
        add("c5", "runhealth", "control", arm, "runs finished", r.finished, "21 of 21", fin,
            f"{r.run_dirs} run dirs; TRAIN_DONE {r.TRAIN_DONE}, SCORE_DONE {r.SCORE_DONE}, "
            f"LAW_DONE {r.LAW_DONE} of {r.assigned}")
        leak = "PASS" if r.n_runs_train75 == 0 and r.n_runs_extra75 == 0 else "FAIL"
        add("c5", "runhealth", "control", arm, "runs with a 7.5M product in train_pairs",
            r.n_runs_train75, "0", leak, f"read on {int(r.n_configs)} configs; "
            f"runs with a 7.5M product in extra_pairs: {int(r.n_runs_extra75)}")
        ex = "PASS" if r.n_extra_ok == r.n_configs else "FAIL"
        add("c5", "runhealth", "control", arm, "extra_pairs count per track", r.n_extra_ok,
            "2" if arm == "Darm1" else "306 histone / 30 DNase", ex,
            f"{int(r.n_extra_ok)} of {int(r.n_configs)} configs correct; counts {r.extra_counts}")
        nb = int(r.n_best_last10)
        bs = "FAIL" if nb > C5_MAX_LAST10 else ("PASS" if r.TRAIN_DONE == 21 else "NOT-READABLE")
        add("c5", "runhealth", "control", arm, f"runs with best step > {LAST10}", nb,
            f"<= {C5_MAX_LAST10} of 21", bs,
            f"{nb} of {r.TRAIN_DONE} trained runs; best step median {r.best_step_median:.0f} "
            f"(range {r.best_step_min:.0f}-{r.best_step_max:.0f}); at the cap {int(r.n_at_cap)}; "
            f"Dref {int(dref.n_best_last10)} of {int(dref.TRAIN_DONE)}")
        c5[arm] = both([fin, leak, ex, bs])
        add("c5", "runhealth", "control", arm, "arm reading", np.nan, "all items", c5[arm],
            "a FAIL invalidates the run")
    add("c5", "runhealth", "control", "both", "check reading", np.nan, "all 42 runs",
        both(c5.values()), f"Darm1 {c5['Darm1']}, Darm2 {c5['Darm2']}")
    # combination rule
    t = pd.DataFrame(rows)
    get = lambda c: t[(t.check == c) & t.item.isin(["check reading", "gap closed on heldout_r2",
                                                    "heldout_r2 vs trained_r4 + 2 x spread"])
                      ].reading.iloc[-1]
    claims = {c: get(c) for c in ("c1", "c3", "c4")}
    ctrl = {c: get(c) for c in ("c2", "c5")}
    darm1_ctrl = {c: t[(t.check == c) & (t.arm == "Darm1") & (t.item == "arm reading")].reading.iloc[0]
                  for c in ("c2", "c5")}
    claim_all = both(claims.values()) if "NOT-READABLE" not in claims.values() else "NOT-READABLE"
    if "FAIL" in ctrl.values():
        rule = "INVALID (a control fails)"
    elif all(v == "PASS" for v in ctrl.values()):
        rule = claim_all
    else:
        rule = "NOT-READABLE (controls not yet readable)"
    add("rule", "all(c1, c3, c4) on Darm1 | c2, c5 pass", "rule", "Darm1", "rule reading", np.nan,
        "all", rule,
        f"claims c1 {claims['c1']}, c3 {claims['c3']}, c4 {claims['c4']} -> all = {claim_all}; "
        f"controls c2 {ctrl['c2']}, c5 {ctrl['c5']} (Darm1 parts: c2 {darm1_ctrl['c2']}, "
        f"c5 {darm1_ctrl['c5']})")
    return pd.DataFrame(rows), claims, ctrl, claim_all, rule


def darm2_read(m5):
    """Darm2 is read, not checked: Darm2 - Darm1 on heldout_r2 per class, on Darm2's tracks."""
    out = []
    for cls in ("histone", "dnase"):
        sc = "all7"
        a2 = cell(m5, scope=sc, arm="Darm2", group="heldout_r2", **{"class": cls})
        if a2 is None or not a2.complete:
            sc = "darm2_tracks"
            a2 = cell(m5, scope=sc, arm="Darm2", group="heldout_r2", **{"class": cls})
        a1 = cell(m5, scope=sc, arm="Darm1", group="heldout_r2", **{"class": cls})
        if a1 is None or a2 is None:
            continue
        gap = a2.median_abs_err - a1.median_abs_err
        out.append({"class": cls, "scope": sc, "tracks": a2.tracks, "Darm1": a1.median_abs_err,
                    "spread_Darm1": a1.seed_spread, "Darm2": a2.median_abs_err,
                    "spread_Darm2": a2.seed_spread, "Darm2_minus_Darm1": gap,
                    "label": noise_label(gap, a1.seed_spread, a2.seed_spread, True, True)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------- reports


def md_table(t, cols, fmt=f3):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) if isinstance(r[c], (float, np.floating))
                                        else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def readings_md(m5, m1, m3, nt, ck, d2r):
    D = (m5.scope == "all7") & (m5["class"] == "all")
    p = ["# t136 readings — h28, the arm-to-arm training grid", "",
         "Readings, not judged. Nothing here is a tick; the PI ticks h28's boxes.", "",
         "## Runs", "", md_table(nt, [c for c in ("arm", "assigned", "run_dirs", "TRAIN_DONE",
                                                   "SCORE_DONE", "LAW_DONE", "finished",
                                                   "n_runs_train75", "extra_counts",
                                                   "n_best_last10", "best_step_median")
                                     if c in nt]), "",
         "## Checks (readings)", "", md_table(ck, ["check", "name", "arm", "item", "value",
                                                   "threshold", "reading", "detail"]), "",
         "## Depth law, all 7 tracks (median |log2 error|; spread = max - min over seeds)", "",
         md_table(m5[D], ["arm", "group", "status", "median_abs_err", "seed_spread",
                          "mean_seed_medians", "n_runs", "n_rows", "complete", "gap_vs_Dref",
                          "label_vs_Dref", "gap_vs_Dviabase", "label_vs_Dviabase"]), "",
         "## Held-out arm->arm pairs, pooled CRPS, all 7 tracks", ""]
    S = (m1.scope == "all7") & (m1["class"] == "all") & (m1.knob == "all")
    p += [md_table(m1[S], ["arm", "subgroup", "crps_all", "crps_oracle_scaled_all",
                           "scale_error_all", "seed_spread", "n_pairs", "n_runs", "complete",
                           "gap_vs_Dref", "label_vs_Dref", "closed_crps_all",
                           "closed_crps_oracle_scaled_all"]), "",
          "## Trained base<->arm and base<->7.5M CRPS, all 7 tracks", "",
          md_table(m3[(m3.scope == "all7") & (m3["class"] == "all")],
                   ["arm", "group", "crps_all", "crps_oracle_scaled_all", "scale_error_all",
                    "seed_spread", "n_runs", "complete", "gap_vs_Dref", "label_vs_Dref"]), "",
          "## Darm2 (read, not checked): Darm2 - Darm1 on heldout_r2", "",
          md_table(d2r, list(d2r.columns)) if len(d2r) else "(no Darm2 track with 3 seeds)", ""]
    return "\n".join(p)


def digest_md(m5, m1, nt, ck, claims, ctrl, claim_all, rule, d2r):
    def row(c, arm, item):
        s = ck[(ck.check == c) & (ck.arm == arm) & (ck.item == item)]
        return s.iloc[0] if len(s) else None

    n1, n2 = (nt[nt.arm == a].iloc[0] for a in GRID)
    c1 = row("c1", "Darm1", "gap closed on heldout_r2")
    c3 = row("c3", "Darm1", "heldout_r2 vs trained_r4 + 2 x spread")
    c4 = row("c4", "Darm1", "check reading")
    L = ["# t136 digest — h28 readings", "",
         "These are readings. They are not judged. The PI ticks the boxes of h28.", "",
         "## What exists", "",
         f"- Darm1 has {n1.finished} of 21 runs finished (trained, scored and law-scored).",
         f"- Darm2 has {n2.finished} of 21 runs finished. {n2.TRAIN_DONE} of 21 runs are trained "
         f"and scored; {n2.LAW_DONE} of 21 have their law scores. The other runs are not finished.",
         "- Dref and Dviabase are the phase-2 reference runs on Fir (21 each). They were read, "
         "not rerun.", "",
         "## What each check reads", "",
         "Spread is the seed spread: max − min over seeds 0, 1 and 2. Every CRPS has its "
         "oracle_scaled and scale_error parts beside it.", "",
         f"- **c1 lawheldout (claim): {c1.reading}.** {c1.detail}. The gap closed is "
         f"{f3(c1.value)}; the threshold is 0.5.",
         f"- **c2 basefit (control).** " + " ".join(
             f"{a}: {row('c2', a, 'arm reading').reading}." for a in GRID) + " " + " ".join(
             f"{r.arm}, {r['item']}: {r.detail}." for _, r in ck[(ck.check == 'c2')
                                                                   & ck.arm.isin(GRID)
                                                                   & (ck['item'] != 'arm reading')].iterrows()),
         f"- **c3 ratiotransfer (claim): {c3.reading}.** {c3.detail}.",
         f"- **c4 crpsheldout (claim): {c4.reading}.** {c4.detail}. " + " ".join(
             f"{r['item']}: {f3(r.value)} against {r.threshold} ({r.reading})."
             for _, r in ck[(ck.check == 'c4') & (ck['item'] != 'check reading')].iterrows())
         + " By class (same pairs): " + " ".join(
             f"{cls}: Dref {f3(x['Dref'].crps_all)}, Dviabase {f3(x['Dviabase'].crps_all)}, Darm1 "
             f"{f3(x['Darm1'].crps_all)} (oracle_scaled {f3(x['Darm1'].crps_oracle_scaled_all)}, "
             f"scale_error {f3(x['Darm1'].scale_error_all)}; spread {f3(x['Darm1'].seed_spread)}), "
             f"gap closed {f3(x['Darm1'].closed_crps_all)}."
             for cls, x in ((c, {a: cell(m1, scope='all7', arm=a, subgroup='heldout75', knob='all',
                                          **{'class': c}) for a in ('Dref', 'Dviabase', 'Darm1')})
                            for c in ('histone', 'dnase'))
             if all(v is not None for v in x.values())),
         f"- **c5 runhealth (control).** " + " ".join(
             f"{r.arm}, {r['item']}: {r.reading} ({r.detail})." for _, r in
             ck[(ck.check == 'c5') & ck.arm.isin(GRID) & (ck['item'] != 'arm reading')].iterrows()),
         "", "## What the combination rule gives", "",
         f"The rule is `all` over c1, c3 and c4 on Darm1. It applies only when c2 and c5 pass.",
         f"The claim checks read: c1 {claims['c1']}, c3 {claims['c3']}, c4 {claims['c4']}. "
         f"So `all` over them reads {claim_all}.",
         f"The controls read: c2 {ctrl['c2']}, c5 {ctrl['c5']}. Thus the rule reads: {rule}.",
         f"If c2 and c5 both pass when Darm2 finishes, the rule reads {claim_all}.",
         "The depth-law values are medians over every row of a cell (all seeds, all tracks), as "
         "for the phase-2 reference values 0.195, 0.125 and 0.052. The mean of the per-seed "
         "medians is in m5_depthlaw_table.tsv (mean_seed_medians).",
         "When Darm2 finishes, the controls become readable. Then the rule reading is final, "
         "unless a control fails.", ""]
    if len(d2r):
        L += ["## Darm2 (read, not checked)", ""]
        for _, r in d2r.iterrows():
            L.append(f"- {r['class']} (tracks {r.tracks}): Darm2 {f3(r.Darm2)} (spread "
                     f"{f3(r.spread_Darm2)}) against Darm1 {f3(r.Darm1)} (spread "
                     f"{f3(r.spread_Darm1)}) on heldout_r2; the difference is "
                     f"{f3(r.Darm2_minus_Darm1)} ({r.label}).")
        L += ["", "The null predicts a Darm2 gain on histone tracks only. This is a reading on "
                  "few tracks and 3 seeds, not a check.", ""]
    return "\n".join(L)


def main(here):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    d, runs = load(here)
    nt = n_table(runs)
    m5 = m5_depthlaw(d)
    m1 = m1_heldout(d)
    m3 = m3_trained(d)
    ck, claims, ctrl, claim_all, rule = checks(m5, m1, m3, nt)
    d2r = darm2_read(m5)
    H = Path(here)
    nt.to_csv(H / "n_table.tsv", sep="\t", index=False)
    m5.to_csv(H / "m5_depthlaw_table.tsv", sep="\t", index=False)
    m1.to_csv(H / "m1_heldout_table.tsv", sep="\t", index=False)
    m3.to_csv(H / "m3_trained_table.tsv", sep="\t", index=False)
    ck.to_csv(H / "checks_readings.tsv", sep="\t", index=False)
    (H / "READINGS.md").write_text(readings_md(m5, m1, m3, nt, ck, d2r))
    (H / "DIGEST.md").write_text(digest_md(m5, m1, nt, ck, claims, ctrl, claim_all, rule, d2r))
    print(nt.to_string(index=False))
    print(ck[["check", "arm", "item", "value", "threshold", "reading"]].to_string(index=False))
    print(f"rule: {rule}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent)
