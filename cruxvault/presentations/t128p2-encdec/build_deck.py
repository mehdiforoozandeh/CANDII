"""Rebuild index.html from deck_src.html by embedding the phase-2 readings as DATA.

Every number on a slide is read here from a results file, never typed by hand (the phase-1 deck's
pattern, ../t128-encdec-sym/build_deck.py: the readings are exploratory and have no metrics.json
addresses, so `crux deck --verify` does not apply). Rebuild after new readings land:

    bash tools/t128p2/readings/run_all.sh
    python cruxvault/presentations/t128p2-encdec/build_deck.py

Inputs: the phase-2 tables written by tools/t128p2/readings/ (gitignored). The default is the main
checkout's cruxvault/results/t128p2/; override with T128P2_RESULTS=<dir>.
  n_table.tsv            runs per arm: assigned and the DONE marks
  runs_fir.tsv           one row per run: steps run, best step, seconds per step (metric 9)
  m1_never_table.tsv     metric 1, never-trained arm -> arm CRPS per knob and subgroup, with its split
  m2_gap_table.tsv       metric 2, never-trained minus trained CRPS, same arm
  m3_trained_table.tsv   metric 3, trained-pair CRPS
  m4_heldout_table.tsv   metric 4, held-out 7.5M depth CRPS, both directions
  m5_depthlaw_table.tsv  metric 5, depth-law median |log2 error|
  m6_hop1_table.tsv      metric 6, hop-1 CRPS: the prediction of the base against the real base
  m7_agree_table.tsv     metric 7, middle consistency |Z - Z'| against |x - x'|
  m8_swap_table.tsv, m8_swap_points.tsv.gz   metric 8, swap check against the identity distance
  m9_steps_table.tsv     metric 9, best step in the last 10 % of the cap
  rank_table.tsv         arms ranked by metric 1, metric 2 beside
  cclean_table.tsv       the middle setting: how far it moved from its start
  pair_table.tsv.gz      one row per scored pair (used for the metric-2 pool check only)
  phase1_knob_table.tsv  phase 1 (4000 steps), copied by the readings, never recomputed

NAMES maps every arm code and g version to its plain name before anything enters DATA; the build
fails if a code leaks. Stdlib only.

    /Users/mforooz/miniforge3/envs/candii/bin/python cruxvault/presentations/t128p2-encdec/build_deck.py
"""
import collections, csv, gzip, json, math, os, pathlib, re, statistics as st

HERE = pathlib.Path(__file__).resolve().parent
R = pathlib.Path(os.environ.get(
    "T128P2_RESULTS", "/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results/t128p2"))
NAMES = {"DSfree": "learned middle setting (phase-1 default)",
         "DSfixed": "middle fixed at the base's setting",
         "DSanchor": "anchored middle: the encoder predicts the base track",
         "D": "old rung D retrained, 12,000 steps",
         "Dviabase": "old rung D through the base",
         "per_track": "one g per track", "across": "one g across tracks"}
P1 = {"DS": "DSfree", "D": "D", "Dviabase": "Dviabase"}      # phase-1 codes of the same arms
CODES = re.compile(r"DSfree|DSfixed|DSanchor|Dviabase|DSANC")


def tsv(name):
    p = R / name
    op = gzip.open(p, "rt") if name.endswith(".gz") else open(p)
    with op as f:
        return list(csv.DictReader(f, delimiter="\t"))


def num(v, nd=5):
    if v is None or v == "" or str(v).lower() in ("nan", "none", "<na>"):
        return None
    f = float(v)
    return None if math.isnan(f) else round(f, nd)


def nm(code):
    return NAMES[code]


def seeds(r, stem):
    return [num(r.get(f"{stem}_s{s}")) for s in (0, 1, 2) if num(r.get(f"{stem}_s{s}")) is not None]


def crps(r, pre=""):
    """[CRPS, oracle-scaled, scale error] of a row; the split always travels with the CRPS."""
    return [num(r[f"{pre}crps_all"]), num(r[f"{pre}crps_oracle_scaled_all"]), num(r[f"{pre}scale_error_all"])]


# ---------------- run counts ----------------
nt = tsv("n_table.tsv")
runs = tsv("runs_fir.tsv")
n = {nm(r["arm"]): {"assigned": int(float(r["assigned"])),
                    "done": int(float(r["LAW_DONE"] or r["viabase_LAW_DONE"] or 0))} for r in nt}
counts = {"runs": sum(v["assigned"] for k, v in n.items() if k != nm("Dviabase")),
          "done": sum(v["done"] for k, v in n.items() if k != nm("Dviabase")),
          "g_per_track": len({r["g"] for r in runs if r["g_version"] == "per_track"}),
          "seeds": len({r["seed"] for r in runs}),
          "max_steps": max(int(float(r["max_steps"])) for r in runs),
          "arms": len({r["arm"] for r in runs})}

# ---------------- metric 1: never-trained CRPS ----------------
m1 = [[nm(r["arm"]), nm(r["g_version"]), r["knob"], r["subgroup"]] + crps(r)
      + [num(r["seed_spread"]), seeds(r, "crps"), int(float(r["n_runs"])), int(float(r["n_pairs"])),
         num(r["gap_vs_D"]), r["label_vs_D"], num(r["gap_vs_Dviabase"]), r["label_vs_Dviabase"]]
      for r in tsv("m1_never_table.tsv")]
# phase 1 (4000 steps): the same arms' rows, never and trained, knob all, with spread and seeds
p1 = [[nm(P1[r["arm"]]), nm(r["g_version"]), r["group"]] + crps(r) + [num(r["seed_spread"]),
      [num(r[f"seed{s}"]) for s in (0, 1, 2) if num(r[f"seed{s}"]) is not None]]
      for r in tsv("phase1_knob_table.tsv") if r["arm"] in P1 and r["knob"] == "all"
      and r["group"] in ("never", "trained")]

# ---------------- metric 2: generalisation gap ----------------
m2 = [[nm(r["arm"]), nm(r["g_version"]), r["knob"], num(r["crps_never"]), num(r["crps_trained"]),
       num(r["gen_gap"]), num(r["gen_gap_crps_oracle_scaled_all"]), num(r["gen_gap_scale_error_all"]),
       num(r["seed_spread"]), seeds(r, "gen_gap"), int(float(r["n_pairs_never"])), int(float(r["n_pairs_trained"])),
       num(r["gap_vs_D"]), r["label_vs_D"]] for r in tsv("m2_gap_table.tsv")]

# why the pooled gap is negative: which pairs sit in each pool (simple means over the scored pairs)
pairs = tsv("pair_table.tsv.gz")
pool = collections.defaultdict(lambda: collections.defaultdict(list))
for r in pairs:
    if r["group"] in ("same", "trained"):
        key = (r["arm"], r["g_version"])
        c = float(r["crps_all"])
        pool[key][(r["group"], r["knob"])].append(c)
        if r["knob"] == "depth":
            tgt = "base" if r["arm_tgt"] == "base" else ("arm" if r["group"] == "trained" else r["level_tgt"])
            pool[key][(r["group"] + "_depth_tgt", tgt)].append(c)
mix = []
for (arm, gv), d in sorted(pool.items()):
    same = {k: v for (g, k), v in d.items() if g == "same"}
    trn = {k: v for (g, k), v in d.items() if g == "trained"}
    if not trn:                                   # via the base has no trained pairs of its own
        continue
    w = {k: len(v) for k, v in same.items()}
    rew = sum(st.mean(trn[k]) * w[k] for k in w) / sum(w.values())
    s_all = st.mean([c for v in same.values() for c in v])
    mix.append({"arm": nm(arm), "g": nm(gv), "never_mean": round(s_all, 4),
                "trained_mean": round(st.mean([c for v in trn.values() for c in v]), 4),
                "trained_reweighted": round(rew, 4), "gap_reweighted": round(s_all - rew, 4),
                "knobs_trained_only": {k: [len(v), round(st.mean(v), 4)] for k, v in trn.items() if k not in same},
                "n_never": {k: len(v) for k, v in same.items()}, "n_trained": {k: len(v) for k, v in trn.items()},
                "depth_trained_by_target": {k: [len(v), round(st.mean(v), 4)] for (g, k), v in d.items()
                                            if g == "trained_depth_tgt"},
                "depth_never_by_target": {k: [len(v), round(st.mean(v), 4)] for (g, k), v in d.items()
                                          if g == "same_depth_tgt"}})

# ---------------- metrics 3-6 ----------------
m3 = [[nm(r["arm"]), nm(r["g_version"]), r["knob"]] + crps(r)
      + [num(r["seed_spread"]), seeds(r, "crps"), int(float(r["n_runs"])), int(float(r["n_pairs"])),
         num(r["gap_vs_D"]), r["label_vs_D"], num(r["phase1_4000steps_crps_all"])]
      for r in tsv("m3_trained_table.tsv") if r["knob"] == "all"]
m4 = [[nm(r["arm"]), nm(r["g_version"]), r["direction"]] + crps(r)
      + [num(r["seed_spread"]), seeds(r, "crps"), int(float(r["n_runs"])), int(float(r["n_pairs"])),
         num(r["gap_vs_D"]), r["label_vs_D"]] for r in tsv("m4_heldout_table.tsv")]
m5 = [[nm(r["arm"]), nm(r["g_version"]), r["group"], num(r["median_abs_err"]), num(r["seed_spread"]),
       seeds(r, "median_abs_err"), int(float(r["n_rows"])), num(r["gap_vs_D"]), r["label_vs_D"],
       num(r["gap_vs_Dviabase"]), r["label_vs_Dviabase"], num(r["phase1_4000steps_median_abs_err"])]
      for r in tsv("m5_depthlaw_table.tsv")]
SRC = {"decoder arm->base (trained record)": "decoder", "encoder NB head (hop1.json)": "encoder"}
m6 = [[nm(r["arm"]), nm(r["g_version"]), SRC[r["source"]], r["group"]] + crps(r)
      + [num(r["seed_spread"]), seeds(r, "crps"), int(float(r["n_pairs"])), num(r["gap_vs_D"]), r["label_vs_D"]]
      for r in tsv("m6_hop1_table.tsv") if r["knob"] == "all"]

# ---------------- metric 7: middle consistency ----------------
m7 = [[nm(r["arm"]), nm(r["g_version"]), int(r["n_knobs_differ"]), num(r["median_abs_z_diff"]),
       num(r["seed_spread"]), num(r["median_abs_x_diff"]), num(r["median_ratio_z_over_x"]),
       num(r["median_ratio_z_over_x_seed_spread"]), seeds(r, "median_ratio_z_over_x"), int(float(r["n_rows"]))]
      for r in tsv("m7_agree_table.tsv")]

# ---------------- metric 8: swap check against identity distance ----------------
m8 = [[nm(r["arm"]), nm(r["g_version"]), num(r["swap_median_abs_log_ratio"]), num(r["seed_spread"]),
       num(r["identity_median_abs_log_ratio"]), num(r["ident_seed_spread"]), int(float(r["n_runs"])),
       num(r["gap_vs_D"]), r["label_vs_D"]] for r in tsv("m8_swap_table.tsv") if r["knob"] == "all"]
sp = collections.defaultdict(lambda: ([], []))          # one point per run: median over its knobs
for r in tsv("m8_swap_points.tsv.gz"):
    key = (r["arm"], r["g_version"], r["seed"], r["track"])
    a, b = num(r["swap_median_abs_log_ratio"]), num(r["identity_median_abs_log_ratio"])
    if a is not None:
        sp[key][0].append(a)
    if b is not None:
        sp[key][1].append(b)
m8_pts = [[nm(k[0]), nm(k[1]), round(st.median(v[1]), 4) if v[1] else None, round(st.median(v[0]), 4)]
          for k, v in sp.items() if v[0]]

# ---------------- metric 9: steps ----------------
m9 = [[nm(r["arm"]), nm(r["g_version"]), int(r["n_runs"]), int(float(r["max_steps"])), int(float(r["n_best_last10"])),
       int(float(r["n_at_cap"])), num(r["best_step_median"]), num(r["steps_run_median"]), num(r["train_hours_median"], 2)]
      for r in tsv("m9_steps_table.tsv")]
m9_runs = [[nm(r["arm"]), nm(r["g_version"]), int(float(r["best_step"])), int(float(r["steps_run"]))]
           for r in runs if r["TRAIN_DONE"] == "1"]

# ---------------- ranking ----------------
rank = [[nm(r["arm"]), nm(r["g_version"]), int(r["rank"]), num(r["m1_crps_never"]), num(r["m1_crps_oracle_scaled_all"]),
         num(r["m1_scale_error_all"]), num(r["m1_seed_spread"]), r["m1_label_vs_next"],
         num(r["m1_gap_vs_next"]), num(r["m2_gen_gap"]), num(r["m2_gen_gap_oracle_scaled_all"]),
         num(r["m2_gen_gap_scale_error_all"]), num(r["m2_seed_spread"]), r["m1_label_vs_D"], r["m1_label_vs_Dviabase"]]
        for r in tsv("rank_table.tsv")]

# ---------------- the middle setting: drift from its start ----------------
dr = collections.defaultdict(list)
for r in tsv("cclean_table.tsv"):
    if num(r["drift_from_init"]) is not None:
        dr[(nm(r["arm"]), nm(r["g_version"]))].append(num(r["drift_from_init"]))
cdrift = [[a, g, round(st.median(v), 4), round(max(v), 4)] for (a, g), v in sorted(dr.items())]

data = {"counts": counts, "n": n, "m1": m1, "p1": p1, "m2": m2, "mix": mix, "m3": m3, "m4": m4, "m5": m5,
        "m6": m6, "m7": m7, "m8": m8, "m8_pts": m8_pts, "m9": m9, "m9_runs": m9_runs, "rank": rank,
        "cdrift": cdrift}
blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
assert not CODES.search(blob), "arm code leaked into DATA: " + CODES.search(blob).group(0)
src = (HERE / "deck_src.html").read_text(encoding="utf-8")
assert src.count("__DATA__") == 1
out = src.replace("__DATA__", blob)
assert not CODES.search(out), "arm code in the page: " + CODES.search(out).group(0)
(HERE / "index.html").write_text(out, encoding="utf-8")
print(f"wrote {HERE / 'index.html'} from {R}: {counts['done']} of {counts['runs']} runs scored; "
      f"{len(m9_runs)} runs with steps; via the base {n[nm('Dviabase')]['done']} of {n[nm('Dviabase')]['assigned']}")
