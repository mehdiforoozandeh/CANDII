"""t128 phase 2 readings, second method: stdlib only, straight from the raw run files.

Independent of extract_records.py / analyze.py (no shared code, no pandas). Two steps:

  collect (on Fir): reads scores.json, law.json, hop1.json, TRAIN_DONE, train_log.tsv and
    config.json (not timing.json, which analyze.py uses) of every run under <root>/runs/, and
    law.json of every via-base run under <root>/viabase/; writes the raw values as lists:
      python3 verify_raw.py collect <out.json> <root>
  reduce (on the laptop): computes the headlines
      1. pooled never-trained CRPS per arm x g version (every law pair; knob all), with its
         oracle_scaled / scale_error split: pairs -> tracks -> seeds;
      2. pooled trained-pair CRPS per arm x g version;
      3. depth-law median |log2 error| per arm x g version x group;
      4. hop-1 pooled CRPS of DSanchor's encoder NB head per g version x group;
      5. metric 2 gen_gap, knob all, per arm x g version: per seed, pooled same-knob law pairs
         minus pooled trained pairs; the reading = the mean over seeds;
      6. the noise label of DSfree against D on metric 1 (never, knob all) per g version:
         "incomplete (n_seeds=k)" below 3 seeds on either side, else "counts" when
         |gap| > 2 x the larger seed spread (max - min over seeds), else "within noise";
      7. metric 9 counts per arm x g version: runs with TRAIN_DONE, best step in the last 10 %
         of the run's step cap (TRAIN_DONE's best_step), runs at the cap (the last step of
         train_log.tsv), and the share.
      python3 verify_raw.py reduce <verify_raw.json> <collect.json>
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict


def collect(out, root):
    c = {k: defaultdict(list) for k in ("depth", "trained", "never", "never_os", "never_se",
                                        "same", "hop1", "steps")}
    runs, via = os.path.join(root, "runs"), os.path.join(root, "viabase")
    dirs = [(n.split("_")[0], os.path.join(runs, n)) for n in sorted(os.listdir(runs))]
    if os.path.isdir(via):
        dirs += [("Dviabase", os.path.join(via, n)) for n in sorted(os.listdir(via))]
    n = 0
    for arm, d in dirs:
        if not os.path.isfile(os.path.join(d, "LAW_DONE")):
            continue
        n += 1
        files = ["law.json"] if arm == "Dviabase" else ["scores.json", "law.json"]
        if arm != "Dviabase" and os.path.isfile(os.path.join(d, "HOP1_DONE")):
            files.append("hop1.json")
        for f in files:
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
                    if r["arm_src"] == r["arm_tgt"]:
                        c["same"][key].append(r["crps_all"])
                elif k == "hop1":
                    c["hop1"][f"{arm}|{gv}|{r['group']}|{seed}|{r['track']}"].append(r["crps_all"])
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
        c["steps"][f"{name.split('_')[0]}|{gv}"].append([best, last, cfg["max_steps"]])
    json.dump(c, open(out, "w"))
    print(f"collect: {n} runs with LAW_DONE -> {out}")


def per_seed(d):
    """{arm|gv|seed|track: [crps]} -> {arm|gv: {seed: mean over tracks of mean over pairs}}."""
    out = defaultdict(lambda: defaultdict(list))
    for k, v in d.items():
        arm, gv, seed, _ = k.split("|")
        out[f"{arm}|{gv}"][seed].append(st.mean(v))
    return {k: {sd: st.mean(t) for sd, t in s.items()} for k, s in out.items()}


def label(a, b):
    """Noise label of the gap a - b, each a {seed: value} dict."""
    k = min(len(a), len(b))
    if k < 3:
        return f"incomplete (n_seeds={k})"
    sa, sb = max(a.values()) - min(a.values()), max(b.values()) - min(b.values())
    gap = st.mean(a.values()) - st.mean(b.values())
    return "counts" if abs(gap) > 2 * max(sa, sb) else "within noise"


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
           "trained_crps_pooled": pooled(c["trained"]),
           "depth_median_abs_err": {k: st.median(v) for k, v in sorted(c["depth"].items())},
           "hop1_crps_pooled": pooled(c["hop1"], nkey=3), "m9": {}, "gen_gap_all": {},
           "label_DSfree_vs_D_never": {}}
    same, tr, nv = per_seed(c["same"]), per_seed(c["trained"]), per_seed(c["never"])
    for k in sorted(set(same) & set(tr)):
        gaps = [same[k][sd] - tr[k][sd] for sd in same[k] if sd in tr[k]]
        res["gen_gap_all"][k] = st.mean(gaps)
    for gv in sorted({k.split("|")[1] for k in nv}):
        a, b = nv.get(f"DSfree|{gv}"), nv.get(f"D|{gv}")
        if a and b:
            res["label_DSfree_vs_D_never"][f"DSfree|{gv}"] = label(a, b)
    for k, v in sorted(c["steps"].items()):
        nb = sum(1 for b, _, m in v if b >= 0.9 * m)
        res["m9"][k] = {"n_runs": len(v), "n_best_last10": nb,
                        "n_at_cap": sum(1 for _, s, m in v if s >= m), "share_best_last10": nb / len(v)}
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
