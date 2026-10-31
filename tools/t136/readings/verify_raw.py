"""t136 readings, second method: stdlib only, straight from the raw run files.

Independent of extract_records.py / analyze.py (no shared code, no pandas). It finds pairs by the
records' `knob_combo` string ("depth:7.5M" in it = a held-out pair; config extra_pairs matched by
knob_combo), and the best step by TRAIN_DONE (analyze.py uses ids/levels and timing.json).

  collect (on the cluster):
      python3 verify_raw.py collect <out.json> <root> grid    # Nibi: <root>/runs/*
      python3 verify_raw.py collect <out.json> <root> ref     # Fir: D_<g> runs + viabase (Dref, Dviabase)
  reduce (on the laptop):
      python3 verify_raw.py reduce <verify_raw.json> <collect.json> [<collect.json> ...]
    computes, for scope all7 (every track) and darm2_tracks (the tracks where Darm2 holds 3 seeds):
      m5  depth-law median |log2 error| per arm x group (median over every row; seed spread from
          the per-seed medians), groups trained_base, heldout_base, heldout_r2, trained_r4;
      m1  pooled CRPS (+ oracle_scaled, scale_error) on the law pairs that touch depth 7.5M;
      m3  pooled CRPS of the trained base<->arm pairs;
      n   per grid arm: runs finished, train-pair leaks, extra-pair counts, best step > 10 800;
      the readings of c1-c5 and of the combination rule, from these numbers alone.
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict

SEEDS = ("0", "1", "2")
DNASE = "C12M02"
R4 = {"depth:15M->depth:3.75M", "depth:3.75M->depth:15M"}


def jload(p):
    with open(p) as fh:
        return json.load(fh)


def collect(out, root, mode):
    c = {"depth": defaultdict(list), "held": defaultdict(list), "trained": defaultdict(list),
         "runs": []}
    if mode == "grid":
        src = os.path.join(root, "runs")
        dirs = [(None, os.path.join(src, n), ("scores.json", "law.json")) for n in sorted(os.listdir(src))]
    else:
        dirs = []
        for sub, arm, files in (("runs", "Dref", ("scores.json", "law.json")),
                                ("viabase", "Dviabase", ("law.json",))):
            src = os.path.join(root, sub)
            for n in sorted(os.listdir(src)):
                if n.startswith("D_") and not n.startswith("D_all_") and "_counts_real_s" in n:
                    dirs.append((arm, os.path.join(src, n), files))
    for arm, d, files in dirs:
        cp = os.path.join(d, "config.json")
        cfg = jload(cp) if os.path.isfile(cp) else {}
        if arm is None:
            arm = cfg.get("arm") or os.path.basename(d).split("_")[0]
        extra = {p["knob_combo"] for p in cfg.get("extra_pairs") or []}
        mk = {m: os.path.isfile(os.path.join(d, m)) for m in ("TRAIN_DONE", "SCORE_DONE", "LAW_DONE")}
        info = {"arm": arm, "name": os.path.basename(d), "g": cfg.get("g"), **mk,
                "n_extra": len(cfg.get("extra_pairs") or []) if cfg else None,
                "n_train75": sum("depth:7.5M" in p["knob_combo"] for p in cfg.get("train_pairs") or []),
                "n_extra75": sum("depth:7.5M" in k for k in extra), "best": None}
        if mk["TRAIN_DONE"]:
            w = open(os.path.join(d, "TRAIN_DONE")).read().split()
            info["best"] = int(w[w.index("best_step") + 1])
        c["runs"].append(info)
        for f in files:
            if not mk["SCORE_DONE" if f == "scores.json" else "LAW_DONE"]:
                continue
            s = jload(os.path.join(d, f))
            seed = str(s["run"]["seed"])
            for r in s["records"]:
                if r.get("eval") != "score":
                    continue
                kc, k, tr = r["knob_combo"], r["kind"], r["track"]
                if k == "depthlaw":
                    src_ = r["depth_source"]
                    if src_ == "law":
                        grp = "heldout_r2" if "depth:7.5M" in kc else ("trained_r4" if kc in R4 else None)
                    else:
                        grp = {"trained": "trained_base", "heldout": "heldout_base"}[src_]
                    if grp and r["depth_log2_scale_pred"] is not None:
                        c["depth"][f"{arm}|{grp}|{seed}|{tr}"].append(
                            abs(r["depth_log2_scale_pred"] - r["depth_log2_ratio_true"]))
                elif k == "law" and "depth:7.5M" in kc:
                    c["held"][f"{arm}|{seed}|{tr}"].append(
                        [r["crps_all"], r["crps_oracle_scaled_all"], r["scale_error_all"]])
                elif k == "trained" and r["direction"] != "arm_to_arm":
                    c["trained"][f"{arm}|{seed}|{tr}"].append(r["crps_all"])
    with open(out, "w") as fh:
        json.dump(c, fh)
    print(f"collect {mode}: {len(c['runs'])} run dirs -> {out}")


def tracks_with_3(keys, arm, prefix=""):
    """Tracks with every seed for `arm` among keys 'arm|[prefix]seed|track'."""
    seen = defaultdict(set)
    for k in keys:
        p = k.split("|")
        if p[0] == arm and (not prefix or p[1] == prefix):
            seen[p[-1]].add(p[-2])
    return sorted(t for t, s in seen.items() if set(SEEDS) <= s)


def reduce(out, parts):
    c = {"depth": {}, "held": {}, "trained": {}, "runs": []}
    for p in parts:
        x = jload(p)
        for k in ("depth", "held", "trained"):
            c[k].update(x[k])
        c["runs"] += x["runs"]
    all_tracks = sorted({k.split("|")[-1] for k in c["held"]})
    res = {"m5": {}, "m1": {}, "m3": {}, "n": {}, "readings": {}}
    arms = sorted({k.split("|")[0] for k in c["depth"]} | {k.split("|")[0] for k in c["held"]})
    scopes = {"all7": all_tracks}
    d2 = {"m5": {g: tracks_with_3([k for k in c["depth"] if k.split("|")[1] == g], "Darm2", g)
                 for g in ("trained_base", "heldout_base", "heldout_r2", "trained_r4")},
          "m1": tracks_with_3(c["held"], "Darm2"), "m3": tracks_with_3(c["trained"], "Darm2")}

    def full(keys, arm, trk, grp=None):
        have = defaultdict(set)
        for k in keys:
            p = k.split("|")
            if p[0] == arm and (grp is None or p[1] == grp):
                have[p[-1]].add(p[-2])
        return all(set(SEEDS) <= have.get(t, set()) for t in trk)

    for g in ("trained_base", "heldout_base", "heldout_r2", "trained_r4"):
        for scope, trk in (("all7", all_tracks), ("darm2_tracks", d2["m5"][g])):
            if scope == "darm2_tracks" and (not trk or trk == all_tracks):
                continue
            for arm in arms:
                rows, per = [], defaultdict(list)
                for k, v in c["depth"].items():
                    a, gg, sd, tr = k.split("|")
                    if a == arm and gg == g and tr in trk:
                        rows += v
                        per[sd] += v
                if rows:
                    meds = [st.median(v) for v in per.values()]
                    res["m5"][f"{scope}|{arm}|{g}"] = {
                        "value": st.median(rows), "spread": max(meds) - min(meds) if len(meds) > 1 else None,
                        "complete": full(c["depth"], arm, trk, g)}

    def pool(d, arm, trk, j=None):
        per = defaultdict(list)
        for k, v in d.items():
            a, sd, tr = k.split("|")
            if a == arm and tr in trk:
                per[sd].append(st.mean(x[j] if j is not None else x for x in v))
        if not per:
            return None
        ps = {sd: st.mean(v) for sd, v in per.items()}
        return st.mean(ps.values()), (max(ps.values()) - min(ps.values())) if len(ps) > 1 else None

    for tab, d, cols in (("m1", c["held"], (0, 1, 2)), ("m3", c["trained"], (None,))):
        for scope, trk in (("all7", all_tracks), ("darm2_tracks", d2[tab])):
            if scope == "darm2_tracks" and (not trk or trk == all_tracks):
                continue
            for arm in arms:
                vals = [pool(d, arm, trk, j) for j in cols]
                if vals[0] is None:
                    continue
                rec = {"value": vals[0][0], "spread": vals[0][1], "complete": full(d, arm, trk)}
                if tab == "m1":
                    rec.update(oracle_scaled=vals[1][0], scale_error=vals[2][0])
                res[tab][f"{scope}|{arm}"] = rec

    for arm in ("Darm1", "Darm2"):
        rr = [r for r in c["runs"] if r["arm"] == arm]
        cf = [r for r in rr if r["n_extra"] is not None]
        tr = [r for r in rr if r["TRAIN_DONE"]]
        res["n"][arm] = {
            "run_dirs": len(rr), "finished": sum(r["TRAIN_DONE"] and r["SCORE_DONE"] and r["LAW_DONE"] for r in rr),
            "trained": len(tr), "train75": sum(r["n_train75"] > 0 for r in cf),
            "extra75": sum(r["n_extra75"] > 0 for r in cf),
            "extra_ok": sum(r["n_extra"] == (2 if arm == "Darm1" else (30 if r["g"] == DNASE else 306)) for r in cf),
            "configs": len(cf), "best_last10": sum(r["best"] > 10800 for r in tr)}
    res["n"]["Dref"] = {"best_last10": sum(r["best"] > 10800 for r in c["runs"]
                                           if r["arm"] == "Dref" and r["best"] is not None)}
    res["readings"] = readings(res)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"reduce: {len(res['m5'])} m5 cells, {len(res['m1'])} m1 cells, readings {res['readings']}")


def worst(xs):
    return "FAIL" if "FAIL" in xs else ("NOT-READABLE" if "NOT-READABLE" in xs else "PASS")


def readings(res):
    m5, m1, m3, n = res["m5"], res["m1"], res["m3"], res["n"]
    out = {}
    a, r, v = (m5.get(f"all7|{x}|heldout_r2") for x in ("Darm1", "Dref", "Dviabase"))
    if all(z and z["complete"] for z in (a, r, v)):
        out["c1"] = "PASS" if (r["value"] - a["value"]) / (r["value"] - v["value"]) >= 0.5 else "FAIL"
    else:
        out["c1"] = "NOT-READABLE"
    for arm in ("Darm1", "Darm2"):
        it = []
        for x, y in ((m5.get(f"all7|{arm}|trained_base"), m5.get("all7|Dref|trained_base")),
                     (m3.get(f"all7|{arm}"), m3.get("all7|Dref"))):
            if x and y and x["complete"] and y["complete"]:
                it.append("PASS" if abs(x["value"] - y["value"]) <= 2 * max(x["spread"], y["spread"]) else "FAIL")
            else:
                it.append("NOT-READABLE")
        out[f"c2|{arm}"] = worst(it)
    out["c2"] = worst([out["c2|Darm1"], out["c2|Darm2"]])
    h, t = m5.get("all7|Darm1|heldout_r2"), m5.get("all7|Darm1|trained_r4")
    if h and t and h["complete"] and t["complete"]:
        out["c3"] = "PASS" if h["value"] <= t["value"] + 2 * max(h["spread"], t["spread"]) else "FAIL"
    else:
        out["c3"] = "NOT-READABLE"
    a, r, v = (m1.get(f"all7|{x}") for x in ("Darm1", "Dref", "Dviabase"))
    if all(z and z["complete"] for z in (a, r, v)):
        ok1 = (r["value"] - a["value"]) / (r["value"] - v["value"]) >= 0.5
        ok2 = r["value"] - a["value"] > 2 * max(a["spread"], r["spread"])
        ok3 = ((r["oracle_scaled"] - a["oracle_scaled"]) / (r["oracle_scaled"] - v["oracle_scaled"])) >= 0.25
        out["c4"] = "PASS" if ok1 and ok2 and ok3 else "FAIL"
    else:
        out["c4"] = "NOT-READABLE"
    for arm in ("Darm1", "Darm2"):
        x = n[arm]
        it = ["PASS" if x["finished"] == 21 else "NOT-READABLE",
              "PASS" if x["train75"] == 0 and x["extra75"] == 0 else "FAIL",
              "PASS" if x["extra_ok"] == x["configs"] else "FAIL",
              "FAIL" if x["best_last10"] > 2 else ("PASS" if x["trained"] == 21 else "NOT-READABLE")]
        out[f"c5|{arm}"] = worst(it)
    out["c5"] = worst([out["c5|Darm1"], out["c5|Darm2"]])
    cl = [out["c1"], out["c3"], out["c4"]]
    claim_all = "NOT-READABLE" if "NOT-READABLE" in cl else worst(cl)
    ctrl = [out["c2"], out["c5"]]
    if "FAIL" in ctrl:
        out["rule"] = "INVALID (a control fails)"
    elif ctrl == ["PASS", "PASS"]:
        out["rule"] = claim_all
    else:
        out["rule"] = "NOT-READABLE (controls not yet readable)"
    out["claims_all"] = claim_all
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 4 and a[0] == "collect":
        collect(a[1], a[2], a[3])
    elif len(a) >= 3 and a[0] == "reduce":
        reduce(a[1], a[2:])
    else:
        raise SystemExit(__doc__)
