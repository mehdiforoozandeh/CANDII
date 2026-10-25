"""Rebuild index.html from deck_src.html by embedding the numbers as DATA.

Every number on a slide is read here from a results file, never typed by hand (the t126 deck's
pattern: the readings are exploratory and have no metrics.json addresses, so `crux deck --verify`
does not apply). Every count (runs per setting, seeds, g's) is read from the tables too, so a
rebuild after more runs land needs no edit here or in deck_src.html:

    bash tools/t128/readings/run_all.sh
    python cruxvault/presentations/t128-encdec-sym/build_deck.py

Inputs: cruxvault/results/t128/ (gitignored; run_all.sh writes it in the main checkout and mirrors
it into a worktree's own cruxvault/results/t128/). Override with T128_RESULTS=<dir>.
  n_table.tsv                runs per setting x model x cluster: assigned and the DONE marks
  runs_{fir,nibi}.tsv        one row per assigned run (cluster)
  steps_table.tsv            steps run and the best step per setting (4000 = the cap)
  depth_table.tsv            reading 1, the depth law, with each seed's median
  knob_table.tsv             reading 2, pooled CRPS per knob and pair group, with the CRPS split
  twin_table.tsv             reading 3, gain over the no-covariates twin
  collapse_table.tsv         reading 4, median |Z - x| and corr(Z, x)
  agree_table.tsv            reading 5, median |Z - Z'| and median |x - x'|
  swap_table.tsv, swap_points.tsv.gz   reading 6, the swap check and the identity distance
  lambda_table.tsv, lambda_per_g.tsv, lambda_chosen.json   the lambda rule on chr22
  cclean_table.tsv           the learned clean setting, its start, the base and the form-D box

The arm codes of the run names never enter DATA: KEY maps each to a short neutral key, and the
plain names live in deck_src.html (ARMS). Stdlib only.

    /Users/mforooz/miniforge3/envs/candii/bin/python cruxvault/presentations/t128-encdec-sym/build_deck.py
"""
import collections, csv, gzip, json, math, os, pathlib, re, statistics as st

HERE = pathlib.Path(__file__).resolve().parent
R = pathlib.Path(os.environ.get("T128_RESULTS", HERE.parent.parent / "results" / "t128"))
KEY = {"DS": "def", "DSbox": "formd", "DSrt": "rt", "DSrtvae": "rtvae", "DSl0.1": "l01", "DSl1": "l1",
       "DSl10": "l10", "DSvae": "vae", "D": "oldd", "Dviabase": "via"}
SETTINGS = ["def", "formd", "rt", "rtvae", "l01", "l1", "l10", "vae"]


def tsv(name):
    p = R / name
    op = gzip.open(p, "rt") if name.endswith(".gz") else open(p)
    return list(csv.DictReader(op, delimiter="\t"))


def num(v, nd=5):
    if v is None or v == "" or v.lower() in ("nan", "none"):
        return None
    f = float(v)
    return None if math.isnan(f) else round(f, nd)


def k(code):
    return KEY[code]


def seeds(r):
    return [num(r[c]) for c in ("seed0", "seed1", "seed2") if num(r.get(c)) is not None]


# ---------------- run counts (n per setting, from the tables) ----------------
nt = tsv("n_table.tsv")
n = {}
for r in nt:
    if r["cluster"] in ("both", "t126 records_D", "viabase LAW_DONE"):
        n[f"{k(r['arm'])}|{r['model']}"] = {"assigned": int(float(r["assigned"])),
                                            "law": int(float(r["LAW_DONE"] or 0))}
by_cl = collections.Counter()
for r in nt:
    if r["cluster"] in ("fir", "nibi"):
        by_cl[(r["cluster"], "assigned")] += int(float(r["assigned"]))
        by_cl[(r["cluster"], "law")] += int(float(r["LAW_DONE"] or 0))
runs_rows = []
for c in ("fir", "nibi"):
    runs_rows += tsv(f"runs_{c}.tsv")
t128 = [v for kk, v in n.items() if kk.split("|")[0] in SETTINGS]
counts = {"assigned": sum(v["assigned"] for v in t128), "done": sum(v["law"] for v in t128),
          "fir": [by_cl[("fir", "law")], by_cl[("fir", "assigned")]],
          "nibi": [by_cl[("nibi", "law")], by_cl[("nibi", "assigned")]],
          "g_per_track": len({r["g"] for r in runs_rows if r["g"] != "all"}),
          "seeds": len({r["seed"] for r in runs_rows}),
          "settings_by_cluster": {c: sorted({k(r["arm"]) for r in runs_rows if r["cluster"] == c},
                                            key=SETTINGS.index) for c in ("fir", "nibi")}}
steps = {f"{k(r['arm'])}|{r['model']}": [int(r["n_runs"]), int(r["n_at_cap_4000"]), num(r["steps_median"]),
                                          num(r["best_step_median"])] for r in tsv("steps_table.tsv")}
cap = int(next(c for c in tsv("steps_table.tsv")[0] if c.startswith("n_at_cap_")).split("_")[-1])

# ---------------- reading 1: depth law ----------------
depth = [[k(r["arm"]), r["g_version"], r["group"], num(r["median_abs_err"]), num(r["q25_abs_err"]),
          num(r["q75_abs_err"]), num(r["seed_spread"]), seeds(r), int(r["n_runs"]), int(r["n_rows"])]
         for r in tsv("depth_table.tsv")]

# ---------------- reading 2: CRPS per knob with its split ----------------
KNOBS = ("all", "depth", "crop", "mapq", "abproxy", "cross")
knob = [[k(r["arm"]), r["g_version"], r["group"], r["knob"], num(r["crps_all"]), num(r["crps_oracle_scaled_all"]),
         num(r["scale_error_all"]), num(r["seed_spread"]), int(r["n_runs"]), int(r["n_pair_records"]),
         r["label_vs_D"], seeds(r)]
        for r in tsv("knob_table.tsv") if r["knob"] in KNOBS]

# ---------------- reading 3: gain over the twin ----------------
twin = [[k(r["arm"]), r["g_version"], r["group"], r["knob"], num(r["gain_median"]), num(r["seed_spread"]),
         int(r["n_pairs"]), int(float(r["n_runs_real"])), int(float(r["n_runs_nocov"])), r["label_vs_trained"]]
        for r in tsv("twin_table.tsv") if r["knob"] in KNOBS]

# ---------------- readings 4-6: collapse, agreement, swap vs identity ----------------
collapse = [[k(r["arm"]), r["g_version"], r["model"], num(r["median_abs_z_minus_x"]), num(r["seed_spread"]),
             num(r["corr_z_x"]), num(r["corr_seed_spread"]), num(r["median_sigma_z"]), int(r["n_runs"]), seeds(r)]
            for r in tsv("collapse_table.tsv") if r["knob"] == "all"]
agree = [[k(r["arm"]), r["g_version"], r["model"], r["n_knobs"], num(r["median_abs_z_diff"]), num(r["seed_spread"]),
          num(r["median_abs_x_diff"]), int(r["n_runs"]), seeds(r)] for r in tsv("agree_table.tsv")]
swap = [[k(r["arm"]), r["g_version"], r["model"], num(r["swap_median_abs_log_ratio"]), num(r["seed_spread"]),
         num(r["identity_median_abs_log_ratio"]), num(r["ident_seed_spread"]), int(r["n_runs"] or 0)]
        for r in tsv("swap_table.tsv") if r["knob"] == "all"]
sp = collections.defaultdict(lambda: ([], []))          # one point per run: median over its knobs
for r in tsv("swap_points.tsv.gz"):
    key = (k(r["arm"]), r["g_version"], r["model"], r["seed"], r["track"])
    a, b = num(r["swap_median_abs_log_ratio"]), num(r["identity_median_abs_log_ratio"])
    if a is not None:
        sp[key][0].append(a)
    if b is not None:
        sp[key][1].append(b)
swap_pts = [[kk[0], kk[1], kk[2], round(st.median(v[1]), 4), round(st.median(v[0]), 4)]
            for kk, v in sp.items() if v[0] and v[1]]

# ---------------- lambda ----------------
lam = [[k(r["arm"]), r["g_version"], num(r["lam"]), num(r["val_nll_mean"]), num(r["seed_spread"]), seeds(r),
        r["chosen"] == "True", int(r["n_runs"]), int(r["n_runs_full"]), int(float(r["n_g_lowest"] or 0))]
       for r in tsv("lambda_table.tsv")]
lam_g = [[k(r["arm"]), r["g_version"], r["g"], num(r["val_nll_mean"]), int(r["n_seeds"])] for r in tsv("lambda_per_g.tsv")]
lc = json.loads(re.sub(r"\bNaN\b", "null", (R / "lambda_chosen.json").read_text()))
lam_chosen = {gv: {"arm": k(v["arm"]), "lam": v["lam"], "complete": v["complete"],
                   "runner_up": k(v["runner_up"]) if v.get("runner_up") else None,
                   "gap": v.get("gap_to_runner_up"), "label": v.get("label")} for gv, v in lc.items()}

# ---------------- C_clean ----------------
cc = tsv("cclean_table.tsv")
drift = collections.defaultdict(lambda: collections.defaultdict(list))
for r in cc:
    key = f"{k(r['arm'])}|{r['g_version']}|{r['model']}"
    for f in ("drift_from_init", "drift_from_base", "seed_spread", "frac_at_box_edge"):
        if num(r[f]) is not None:
            drift[key][f].append(num(r[f]))
cdrift = {kk: {f: round(st.median(v), 4) for f, v in d.items()} for kk, d in drift.items()}
cdrift = {kk: dict(v, n_runs=max(int(r["n_runs"]) for r in cc if f"{k(r['arm'])}|{r['g_version']}|{r['model']}" == kk))
          for kk, v in cdrift.items()}
# per knob, one g per track, median over the tracks: start, learned, base and the box (default and form D)
pk = collections.defaultdict(lambda: collections.defaultdict(list))
for r in cc:
    if r["g_version"] == "per_track" and r["model"] == "real" and r["arm"] in ("DS", "DSbox"):
        for f in ("init", "learned", "base", "box_lo", "box_hi", "seed_spread"):
            if num(r[f]) is not None:
                pk[(k(r["arm"]), int(r["knob_i"]), r["knob"])][f].append(num(r[f]))
cknob = [[a, i, name] + [round(st.median(d[f]), 4) if d[f] else None
                         for f in ("init", "learned", "base", "box_lo", "box_hi", "seed_spread")]
         for (a, i, name), d in sorted(pk.items(), key=lambda x: (x[0][0], x[0][1]))]

data = {"counts": counts, "n": n, "steps": steps, "cap": cap, "depth": depth, "knob": knob, "twin": twin,
        "collapse": collapse, "agree": agree, "swap": swap, "swap_pts": swap_pts, "lam": lam, "lam_g": lam_g,
        "lam_chosen": lam_chosen, "cdrift": cdrift, "cknob": cknob}
blob = json.dumps(data, separators=(",", ":"))
assert not re.search(r"DSbox|DSrtvae|DSrt\b|DSl0\.1|DSl1\b|DSl10|DSvae|Dviabase|\"DS\"", blob), "arm code leaked"
src = open(HERE / "deck_src.html").read()
assert src.count("__DATA__") == 1
open(HERE / "index.html", "w").write(src.replace("__DATA__", blob))
print(f"wrote {HERE / 'index.html'} from {R}: {counts['done']} of {counts['assigned']} runs with LAW_DONE "
      f"(Fir {counts['fir'][0]}/{counts['fir'][1]}, Nibi {counts['nibi'][0]}/{counts['nibi'][1]}); "
      + ", ".join(f"{s} {n.get(s + '|real', {}).get('law', 0)}" for s in SETTINGS))
