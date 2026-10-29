"""Rebuild index.html from deck_src.html by embedding the phase-3 readings as DATA.

Phase 3 of the pivot study (crux task t134): a bigger g and a bigger f for old rung D, read through
the base. Spec: plan/T126_ENCDEC_D2.md § "Phase 3: PI decisions, 2026-10-27" and its "Design check".
Every number on a slide is read here from a results file, never typed by hand (the phase-2 deck's
mechanism, ../t128p2-encdec/build_deck.py: the readings are exploratory and have no metrics.json
addresses). Rebuild after new readings land:

    bash tools/t128p3/readings/run_all.sh
    python cruxvault/presentations/t128p3-scale/build_deck.py

Inputs: the phase-3 tables written by tools/t128p3/readings/ (gitignored). The default is the main
checkout's cruxvault/results/t128p3/; override with T128P3_RESULTS=<dir>.
  n_table.tsv                runs per arm: assigned and the DONE marks
  runs_fir.tsv               one row per phase-3 run: sizes (g hidden, f channels, n_theta), steps
  ref_runs_fir.tsv           one row per phase-2 run (the reference rows are old rung D's)
  m1_never_table.tsv         metric 1, never-trained CRPS per knob and subgroup, with its split,
                             the noise labels and the pre-registered threshold and reading
  m2_gap_table.tsv           metric 2, never-trained minus trained CRPS, same arm
  m3_trained_table.tsv       metric 3, trained-pair CRPS
  m4_heldout_table.tsv       metric 4, held-out 7.5M depth CRPS, both directions
  m5_depthlaw_table.tsv      metric 5, depth-law median |log2 error|
  m9_steps_table.tsv         metric 9, best steps against the 15,000-step cap
  step_confounded_runs.tsv   runs whose best step is above 12,000 (listed apart)
  rank_table.tsv             the three arms read through the base, ranked by metric 1
  checks.txt, verify.txt     the readings' own checks and the second-method comparison

NAMES maps every arm code and g version to its plain name before anything enters DATA; the build
fails if a code leaks into DATA or into the page. Stdlib only.

    /Users/mforooz/miniforge3/envs/candii/bin/python cruxvault/presentations/t128p3-scale/build_deck.py
"""
import csv, gzip, json, math, os, pathlib, re

HERE = pathlib.Path(__file__).resolve().parent
R = pathlib.Path(os.environ.get(
    "T128P3_RESULTS", "/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results/t128p3"))
NAMES = {"Dbigg": "bigger g (hidden width 256), direct",
         "Dbigf": "bigger f (64 channels), direct",
         "Dbiggviabase": "bigger g, through the base",
         "Dbigfviabase": "bigger f, through the base",
         "D": "old rung D, phase 2 (reference, direct)",
         "Dviabase": "old rung D through the base, phase 2 (reference)",
         "per_track": "one g per track", "across": "one g across tracks"}
ORDER = ["Dbiggviabase", "Dbigfviabase", "Dviabase", "Dbigg", "Dbigf", "D"]
CODES = re.compile(r"Dbigg|Dbigf|Dviabase|viabase|DSfree|DSfixed|DSanchor")
# g is an MLP: 2 x 20 covariates in, two hidden layers, n_theta out (spec: "MLP 40 -> h -> h -> n_theta")
G_IN = 40
# the reference's sizes are phase 2's defaults (model.py HIDDEN 64, form_d.py CHANNELS 32, n_theta 256);
# phase 2's runs table does not record them
REF_SIZES = (64, 32, 256)


def tsv(name):
    p = R / name
    op = gzip.open(p, "rt") if name.endswith(".gz") else open(p)
    with op as f:
        return list(csv.DictReader(f, delimiter="\t"))


def num(v, nd=5):
    if v is None or v == "" or str(v).lower() in ("nan", "none", "<na>", "n/a"):
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


def g_params(h, n_theta):
    return G_IN * h + h + h * h + h + h * n_theta + n_theta


# ---------------- runs and sizes ----------------
nt = tsv("n_table.tsv")
runs = tsv("runs_fir.tsv")
ref_runs = [r for r in tsv("ref_runs_fir.tsv") if r["arm"] == "D"]
n = {nm(r["arm"]): {"assigned": int(float(r["assigned"])),
                    "done": int(float(r["LAW_DONE"] or r["viabase_LAW_DONE"] or 0))} for r in nt}
sizes = {}
for code in ("Dbigg", "Dbigf"):
    s = {(int(r["g_hidden"]), int(r["f_channels"]), int(r["n_theta"])) for r in runs if r["arm"] == code}
    assert len(s) == 1, (code, s)
    sizes[code] = s.pop()
sizes["D"] = REF_SIZES
size = {nm(c): {"g_hidden": h, "f_channels": fc, "n_theta": th, "g_params": g_params(h, th)}
        for c, (h, fc, th) in sizes.items()}
new = [r for r in runs if r["arm"] in ("Dbigg", "Dbigf")]
counts = {"runs": len(new),
          "done": sum(1 for r in new if r["LAW_DONE"] == "1" and r["viabase_LAW_DONE"] == "1"),
          "g_per_track": len({r["g"] for r in new if r["g_version"] == "per_track"}),
          "seeds": len({r["seed"] for r in new}),
          "max_steps": max(int(float(r["max_steps"])) for r in new),
          "ref_max_steps": max(int(float(r["max_steps"])) for r in ref_runs),
          "ref_runs": len(ref_runs),
          "arms": len({r["arm"] for r in new})}

# ---------------- metric 1: never-trained CRPS (through the base and direct) ----------------
m1 = [[nm(r["arm"]), nm(r["g_version"]), r["knob"], r["subgroup"]] + crps(r)
      + [num(r["seed_spread"]), seeds(r, "crps"), int(float(r["n_runs"])), int(float(r["n_pairs"])),
         num(r["gap_vs_D"]), r["label_vs_D"], num(r["gap_vs_Dviabase"]), r["label_vs_Dviabase"],
         num(r["prereg_threshold"]), r["prereg_label"], int(float(r["n_step_confounded"] or 0))]
      for r in tsv("m1_never_table.tsv")]

# ---------------- metrics 2-5 ----------------
m2 = [[nm(r["arm"]), nm(r["g_version"]), r["knob"], num(r["crps_never"]), num(r["crps_trained"]),
       num(r["gen_gap"]), num(r["gen_gap_crps_oracle_scaled_all"]), num(r["gen_gap_scale_error_all"]),
       num(r["seed_spread"]), seeds(r, "gen_gap"), int(float(r["n_pairs_never"])), int(float(r["n_pairs_trained"])),
       num(r["gap_vs_D"]), r["label_vs_D"]] for r in tsv("m2_gap_table.tsv") if r["knob"] == "all"]
m3 = [[nm(r["arm"]), nm(r["g_version"]), r["knob"]] + crps(r)
      + [num(r["seed_spread"]), seeds(r, "crps"), int(float(r["n_runs"])), int(float(r["n_pairs"])),
         num(r["gap_vs_D"]), r["label_vs_D"]]
      for r in tsv("m3_trained_table.tsv") if r["knob"] == "all"]
m4 = [[nm(r["arm"]), nm(r["g_version"]), r["direction"]] + crps(r)
      + [num(r["seed_spread"]), seeds(r, "crps"), int(float(r["n_runs"])), int(float(r["n_pairs"])),
         num(r["gap_vs_D"]), r["label_vs_D"]] for r in tsv("m4_heldout_table.tsv")]
m5 = [[nm(r["arm"]), nm(r["g_version"]), r["group"], num(r["median_abs_err"]), num(r["seed_spread"]),
       seeds(r, "median_abs_err"), int(float(r["n_rows"])), num(r["gap_vs_D"]), r["label_vs_D"],
       num(r["gap_vs_Dviabase"]), r["label_vs_Dviabase"]]
      for r in tsv("m5_depthlaw_table.tsv")]

# ---------------- metric 9: steps ----------------
m9 = [[nm(r["arm"]), nm(r["g_version"]), int(r["n_runs"]), int(float(r["max_steps"])), int(float(r["n_best_last10"])),
       int(float(r["n_at_cap"])), num(r["best_step_median"]), num(r["steps_run_median"]), num(r["train_hours_median"], 2),
       int(float(r["n_best_gt12000"])) if r.get("n_best_gt12000") not in (None, "") else None,
       int(float(r["best_step_min"])), int(float(r["best_step_max"])), num(r["s_per_step_median"], 2)]
      for r in tsv("m9_steps_table.tsv")]
m9_runs = [[nm(r["arm"]), nm(r["g_version"]), int(float(r["best_step"])), int(float(r["steps_run"]))]
           for r in new + ref_runs if r["TRAIN_DONE"] == "1"]
stepconf = [[nm(r["arm"]), r["g"], int(r["seed"]), int(float(r["best_step"])), int(float(r["steps_run"]))]
            for r in tsv("step_confounded_runs.tsv")]

# ---------------- ranking ----------------
rank = [[nm(r["arm"]), nm(r["g_version"]), int(r["rank"]), num(r["m1_crps_never"]), num(r["m1_crps_oracle_scaled_all"]),
         num(r["m1_scale_error_all"]), num(r["m1_seed_spread"]), num(r["m1_gap_vs_Dviabase"]), r["m1_label_vs_Dviabase"],
         num(r["prereg_threshold"]), r["prereg_label"], nm(r["direct_arm"]), num(r["direct_m1_crps_never"]),
         num(r["direct_m1_crps_oracle_scaled_all"]), num(r["direct_m1_scale_error_all"]), num(r["direct_m1_seed_spread"]),
         r["direct_m1_label_vs_D"]]
        for r in tsv("rank_table.tsv")]

# ---------------- the readings' own checks ----------------
chk = (R / "checks.txt").read_text().splitlines()
ver = (R / "verify.txt").read_text().splitlines()
OPT = re.compile(r"^INFO optimisation-failure reading (\S+) (\S+): .*trained crps gap vs D (\S+) \(([^)]*)\) -> (.*)$")
optfail = []
for line in chk:
    m = OPT.match(line)
    if m:
        optfail.append([nm(m.group(1)), nm(m.group(2)), num(m.group(3)), m.group(4),
                        not m.group(5).startswith("not ")])
checks = {"pass": sum(l.startswith("PASS") for l in chk), "fail": sum(l.startswith("FAIL") for l in chk),
          "agree": sum(l.startswith("AGREE") for l in ver), "disagree": sum(l.startswith("DISAGREE") for l in ver)}
assert checks["fail"] == 0 and checks["disagree"] == 0, checks

data = {"arms": [nm(c) for c in ORDER], "counts": counts, "n": n, "size": size, "m1": m1, "m2": m2,
        "m3": m3, "m4": m4, "m5": m5, "m9": m9, "m9_runs": m9_runs, "stepconf": stepconf, "rank": rank,
        "optfail": optfail, "checks": checks}
blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
assert not CODES.search(blob), "arm code leaked into DATA: " + CODES.search(blob).group(0)
src = (HERE / "deck_src.html").read_text(encoding="utf-8")
assert src.count("__DATA__") == 1
out = src.replace("__DATA__", blob)
assert not CODES.search(out), "arm code in the page: " + CODES.search(out).group(0)
(HERE / "index.html").write_text(out, encoding="utf-8")
print(f"wrote {HERE / 'index.html'} from {R}: {counts['done']} of {counts['runs']} runs scored; "
      f"{len(m9_runs)} runs with steps; {checks['pass']} checks pass, {checks['agree']} second-method agreements")
