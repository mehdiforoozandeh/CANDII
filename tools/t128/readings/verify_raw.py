"""t128 readings, second method: stdlib only, straight from the raw run files.

Independent of extract_records.py / analyze.py (no shared code, no pandas). Two steps:

  collect (on a cluster): reads scores.json, law.json, latent.json and TRAIN_DONE (not
    timing.json, which analyze.py uses) of the runs the index file assigns to this cluster, and
    law.json of every via-base run; writes the raw values the headlines need, as lists:
      python3 verify_raw.py collect <out.json> <runs_dir> <idx_file> [<viabase_dir>]
  reduce (on the laptop): merges the collect files of both clusters and computes
      1. depth law median |log2 error| per arm x g version x group (real model, every seed);
      2. pooled trained-pair CRPS (real): pairs -> tracks -> seeds;
      3. pooled never-trained CRPS (real, every law pair): pairs -> tracks -> seeds;
      4. the median over pairs of the gain over the twin on never-trained pairs (DS, DSrt);
      5. collapse: median over products and seeds of median |Z - x| and of corr(Z, x) (real);
      6. the lambda choice per g version from TRAIN_DONE's val_nll (settings 5-7, real).
      python3 verify_raw.py reduce <verify_raw.json> <collect.json> [<collect.json> ...]
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict

LAMS = ("DSl0.1", "DSl1", "DSl10")


def collect(out, runs, idx_file, via=None):
    c = {"depth": defaultdict(list), "trained": defaultdict(list), "never": defaultdict(list),
         "pair": defaultdict(list), "zx": defaultdict(list), "corr": defaultdict(list),
         "val": defaultdict(list)}
    names = [ln.split()[1] for ln in open(idx_file) if len(ln.split()) == 2]
    dirs = [(n.split("_")[0], os.path.join(runs, n)) for n in names]
    if via:
        dirs += [("Dviabase", os.path.join(via, n)) for n in sorted(os.listdir(via))]
    n = 0
    for arm, d in dirs:
        if not os.path.isfile(os.path.join(d, "LAW_DONE")):
            continue
        n += 1
        files = ["law.json"] if arm == "Dviabase" else ["scores.json", "law.json"]
        for f in files:
            s = json.load(open(os.path.join(d, f)))
            run = s["run"]
            gv, model, seed = run["g_version"], run["model"], run["seed"]
            for r in s["records"]:
                if r.get("eval") != "score":
                    continue
                k = r["kind"]
                if k == "depthlaw" and model == "real":
                    t7 = "7.5M" in (r["level_src"], r["level_tgt"])
                    grp = {"trained": "trained", "heldout": "heldout"}.get(
                        r["depth_source"], "never75" if t7 else "never")
                    if r["depth_log2_scale_pred"] is not None:
                        c["depth"][f"{arm}|{gv}|{grp}"].append(
                            abs(r["depth_log2_scale_pred"] - r["depth_log2_ratio_true"]))
                elif k == "trained" and model == "real":
                    c["trained"][f"{arm}|{gv}|{seed}|{r['track']}"].append(r["crps_all"])
                elif k == "law":
                    if model == "real":
                        c["never"][f"{arm}|{gv}|{seed}|{r['track']}"].append(r["crps_all"])
                    if arm in ("DS", "DSrt"):
                        key = "|".join([arm, gv, r["track"], r["arm_src"], r["level_src"],
                                        r["arm_tgt"], r["level_tgt"], model])
                        c["pair"][key].append(r["crps_all"])
        lj = os.path.join(d, "latent.json")
        if arm != "Dviabase" and os.path.isfile(os.path.join(d, "LATENT_DONE")):
            s = json.load(open(lj))
            if s["run"]["model"] == "real":
                gv = s["run"]["g_version"]
                for r in s["records"]:
                    if r["kind"] == "latent":
                        c["zx"][f"{arm}|{gv}"].append(r["median_abs_z_minus_x"])
                        c["corr"][f"{arm}|{gv}"].append(r["corr_z_x"])
    for name in names:
        arm = name.split("_")[0]
        p = os.path.join(runs, name, "TRAIN_DONE")
        if arm in LAMS and name.split("_")[3] == "real" and os.path.isfile(p):
            txt = open(p).read().split()
            gv = "across" if name.split("_")[1] == "all" else "per_track"
            c["val"][f"{arm}|{gv}"].append(float(txt[txt.index("val_nll") + 1]))
    json.dump(c, open(out, "w"))
    print(f"collect: {n} runs with LAW_DONE -> {out}")


def pooled(d):
    """{arm|gv|seed|track: [crps]} -> {arm|gv: mean over seeds of mean over tracks of mean}."""
    per_seed = defaultdict(lambda: defaultdict(list))
    for k, v in d.items():
        arm, gv, seed, _ = k.split("|")
        per_seed[f"{arm}|{gv}"][seed].append(st.mean(v))
    return {k: st.mean(st.mean(t) for t in s.values()) for k, s in sorted(per_seed.items())}


def reduce(out, parts):
    c = defaultdict(lambda: defaultdict(list))
    for p in parts:
        for kind, d in json.load(open(p)).items():
            for k, v in d.items():
                c[kind][k].extend(v)
    res = {"depth_median_abs_err": {k: st.median(v) for k, v in sorted(c["depth"].items())},
           "trained_crps_pooled": pooled(c["trained"]), "never_crps_pooled": pooled(c["never"]),
           "twin_gain_never_median": {}, "collapse_median_abs_z_minus_x": {},
           "collapse_median_corr": {}, "val_nll_mean": {}, "lambda": {}}
    # pair gain: seed-mean CRPS per model, then the median over pairs
    by = defaultdict(dict)
    for k, v in c["pair"].items():
        *pk, model = k.split("|")
        by["|".join(pk)][model] = st.mean(v)
    gains = defaultdict(list)
    for pk, m in by.items():
        if "real" in m and "nocov" in m:
            arm, gv = pk.split("|")[:2]
            gains[f"{arm}|{gv}"].append((m["nocov"] - m["real"]) / abs(m["nocov"]))
    res["twin_gain_never_median"] = {k: st.median(v) for k, v in sorted(gains.items())}
    res["collapse_median_abs_z_minus_x"] = {k: st.median(v) for k, v in sorted(c["zx"].items())}
    res["collapse_median_corr"] = {k: st.median(v) for k, v in sorted(c["corr"].items())}
    res["val_nll_mean"] = {k: st.mean(v) for k, v in sorted(c["val"].items())}
    for gv in ("per_track", "across"):
        m = {a: st.mean(c["val"][f"{a}|{gv}"]) for a in LAMS if c["val"].get(f"{a}|{gv}")}
        res["lambda"][gv] = min(m, key=m.get) if m else None
    json.dump(res, open(out, "w"), indent=1)
    print(f"reduce: lambda {res['lambda']}; {len(res['depth_median_abs_err'])} depth cells")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "collect":
        collect(a[1], a[2], a[3], a[4] if len(a) > 4 else None)
    elif a[0] == "reduce":
        reduce(a[1], a[2:])
    else:
        raise SystemExit(__doc__)
