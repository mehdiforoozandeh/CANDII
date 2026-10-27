"""t128 phase 3 readings, second method: stdlib only, straight from the raw run files.

Independent of extract_records.py / analyze.py (no shared code, no pandas). Two steps:

  collect (on Fir): reads scores.json, law.json, TRAIN_DONE, train_log.tsv and config.json (not
    timing.json, which analyze.py uses) of every run under <root>/runs/, and viabase.json +
    law.json of every via-base dir under <root>/viabase/ (arm = the old run's arm + "viabase");
    writes the raw values as lists:
      python3 verify_raw.py collect <out.json> <root>
  reduce (on the laptop): computes the headlines, phase-3 arms only (the reference is phase 2's
    and was verified there):
      1. pooled never-trained CRPS per arm x g version (every law pair; knob all), direct and via
         the base, with its oracle_scaled / scale_error split: pairs -> tracks -> seeds;
      2. the via-base never-trained pair count per arm x g version (distinct pairs);
      3. pooled trained-pair CRPS per direct arm x g version;
      4. depth-law median |log2 error| per arm x g version x group;
      5. metric 9 per direct arm x g version: runs with TRAIN_DONE, best step in the last 10 % of
         the run's step cap (TRAIN_DONE's best_step), runs at the cap (the last step of
         train_log.tsv), runs with best step > 12,000, the share and the best-step median.
      python3 verify_raw.py reduce <verify_raw.json> <collect.json>
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict


def run_arm(d, name):
    p = os.path.join(d, "config.json")
    if os.path.isfile(p):
        a = json.load(open(p)).get("arm")
        if a:
            return a
    return name.split("_")[0]


def collect(out, root):
    c = {k: defaultdict(list) for k in ("depth", "trained", "never", "never_os", "never_se",
                                        "via_pairs", "steps")}
    runs, via = os.path.join(root, "runs"), os.path.join(root, "viabase")
    dirs = [(run_arm(os.path.join(runs, n), n), os.path.join(runs, n), False)
            for n in sorted(os.listdir(runs))]
    if os.path.isdir(via):
        for n in sorted(os.listdir(via)):
            vj = os.path.join(via, n, "viabase.json")
            old = json.load(open(vj))["old_run_dir"] if os.path.isfile(vj) else os.path.join(runs, n)
            name = os.path.basename(os.path.normpath(old))
            od = old if os.path.isdir(old) else os.path.join(runs, name)
            dirs.append((run_arm(od, name) + "viabase", os.path.join(via, n), True))
    n = 0
    for arm, d, is_via in dirs:
        if not os.path.isfile(os.path.join(d, "LAW_DONE")):
            continue
        n += 1
        for f in (["law.json"] if is_via else ["scores.json", "law.json"]):
            s = json.load(open(os.path.join(d, f)))
            run = s["run"]
            gv, seed = run["g_version"], run["seed"]
            for r in s["records"]:
                if r.get("eval") != "score":
                    continue
                k = r["kind"]
                key = f"{arm}|{gv}|{seed}|{r['track']}"
                if k == "depthlaw":
                    t7 = "7.5M" in (r["level_src"], r["level_tgt"])
                    grp = {"trained": "trained", "heldout": "heldout"}.get(
                        r["depth_source"], "never75" if t7 else "never")
                    if r["depth_log2_scale_pred"] is not None:
                        c["depth"][f"{arm}|{gv}|{grp}"].append(
                            abs(r["depth_log2_scale_pred"] - r["depth_log2_ratio_true"]))
                elif k == "trained":
                    c["trained"][key].append(r["crps_all"])
                elif k == "law":
                    c["never"][key].append(r["crps_all"])
                    c["never_os"][key].append(r["crps_oracle_scaled_all"])
                    c["never_se"][key].append(r["scale_error_all"])
                    if is_via:
                        c["via_pairs"][f"{arm}|{gv}"].append(
                            "|".join(str(r[x]) for x in ("track", "arm_src", "level_src",
                                                         "arm_tgt", "level_tgt")))
    for name in sorted(os.listdir(runs)):
        d = os.path.join(runs, name)
        p = os.path.join(d, "TRAIN_DONE")
        if not os.path.isfile(p):
            continue
        txt = open(p).read().split()
        best = int(txt[txt.index("best_step") + 1])
        rows = [ln.split("\t") for ln in open(os.path.join(d, "train_log.tsv")).read().splitlines()]
        last = int(float(rows[-1][rows[0].index("step")]))
        cfg = json.load(open(os.path.join(d, "config.json")))
        gv = "across" if cfg["g"] == "all" else "per_track"
        c["steps"][f"{run_arm(d, name)}|{gv}"].append([best, last, cfg["max_steps"]])
    json.dump(c, open(out, "w"))
    print(f"collect: {n} run dirs with LAW_DONE -> {out}")


def pooled(d, nkey=2):
    """{k1|..|k_nkey|seed|track: [crps]} -> {k1|..|k_nkey: mean over seeds of mean over tracks of
    mean over pairs}."""
    per_seed = defaultdict(lambda: defaultdict(list))
    for k, v in d.items():
        p = k.split("|")
        per_seed["|".join(p[:nkey])][p[nkey]].append(st.mean(v))
    return {k: st.mean(st.mean(t) for t in s.values()) for k, s in sorted(per_seed.items())}


def reduce(out, part):
    c = json.load(open(part))
    res = {"never_crps_pooled": pooled(c["never"]),
           "never_oracle_scaled_pooled": pooled(c["never_os"]),
           "never_scale_error_pooled": pooled(c["never_se"]),
           "viabase_never_n_pairs": {k: len(set(v)) for k, v in sorted(c["via_pairs"].items())},
           "trained_crps_pooled": pooled(c["trained"]),
           "depth_median_abs_err": {k: st.median(v) for k, v in sorted(c["depth"].items())},
           "m9": {}}
    for k, v in sorted(c["steps"].items()):
        nb = sum(1 for b, _, m in v if b >= 0.9 * m)
        res["m9"][k] = {"n_runs": len(v), "n_best_last10": nb,
                        "n_at_cap": sum(1 for _, s, m in v if s >= m),
                        "n_best_gt12000": sum(1 for b, _, _ in v if b > 12000),
                        "share_best_last10": nb / len(v),
                        "best_step_median": st.median(b for b, _, _ in v)}
    json.dump(res, open(out, "w"), indent=1)
    print(f"reduce: {len(res['never_crps_pooled'])} never cells, {len(res['m9'])} m9 cells")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "collect":
        collect(a[1], a[2])
    elif a and a[0] == "reduce":
        reduce(a[1], a[2])
    else:
        raise SystemExit(__doc__)
