"""Rebuild index.html from deck_src.html by embedding the numbers as DATA.

Every number on a slide is read here from a results file, never typed by hand (the t125 deck's
pattern: the readings are exploratory and have no metrics.json addresses, so `crux deck --verify`
does not apply).

Inputs (all gitignored, local; the t126 tables are identical in the main checkout):
  cruxvault/results/t126/                 the t126 readings (analyze.py on the Nibi extraction)
    depth_table.tsv, depth_points.tsv.gz  depth law per arm x g version x pair group; every point
    knob_table.tsv                        typical-pair gain over the no-covariates twin per knob
    fit_table.tsv                         pooled CRPS on trained pairs, with its split
    ab_table.tsv                          rung-A arms: the learned (a, b) on base<->depth pairs
    lambda_table.tsv, lambda_chosen.json  the lambda rule on chr22
    steps_table.tsv                       steps run per real run (4000 = the cap)
    records_A.tsv.gz, records_D.tsv.gz    per-run records (run counts only)
  cruxvault/results/t126/deck_inputs/     copies, so the build does not depend on /tmp:
    knobreport_depth_table.tsv            t118 knob report: depth law, trained vs never-trained
    knobreport_knob_table.tsv             t118 knob report: gain over the twin per knob
    diag_depthlaw_audit_rows.json         2026-10-19 check: depth law recomputed from count
                                          histograms (field recon) and the old-A (a, b); its script
                                          diag_depthlaw_audit_rows.py
    diag_count_sums.jsonl                 2026-10-19 check: summed counts of every depth product
                                          (script diag_count_sums.py)
    diag_depth_records_row1.tsv           2026-10-19 check: every t118 row-1 depth-law record
                                          (diag_extract_raw.py); composed through the base as in
                                          diag_compose.py

    python cruxvault/presentations/t126-encdec/build_deck.py
"""
import collections, csv, gzip, json, math, pathlib, statistics as st

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent.parent / "results" / "t126"
DI = R / "deck_inputs"
r4 = lambda v: None if v is None or v == "" else round(float(v), 5)
tsv = lambda p: list(csv.DictReader(open(p), delimiter="\t"))
med = st.median

# ---------------- motivation: the t118 knob report ----------------
kd = tsv(DI / "knobreport_depth_table.tsv")
mot_depth = [[r["row"], r["rung"], r["g_version"], r4(r["median_abs_err"])]
             for r in kd if r["depth_source"] == "trained"]
law = {(r["rung"], r["g_version"]): r4(r["median_abs_err"]) for r in kd if r["depth_source"] == "law"}
mot_depth = [x + [law[(x[1], x[2])]] for x in mot_depth]          # [row, design, g, trained, never]
kk = collections.defaultdict(list)
for r in tsv(DI / "knobreport_knob_table.tsv"):
    kk[(r["space"], r["knob"], r["group"])].append(float(r["gain_rel_nocov"]))
mot_knob = {f"{sp}|{k}|{g}": [r4(med(v)), len(v)] for (sp, k, g), v in kk.items()}

# ---------------- diagnosis, 2026-10-19 ----------------
rows = json.load(open(DI / "diag_depthlaw_audit_rows.json"))
sums = [json.loads(l) for l in open(DI / "diag_count_sums.jsonl")]
base = {s["pid"].split("__")[0]: s for s in sums if s["pid"].endswith("base__base")}
sum_err = [abs(math.log2(s["score_sum"] / base[s["pid"].split("__")[0]]["score_sum"])
               - math.log2(s["depth"] / base[s["pid"].split("__")[0]]["depth"]))
           for s in sums if not s["pid"].endswith("base__base")]
q = lambda v: [r4(x) for x in st.quantiles(v, n=4)]
diag = {"audit_n": len(rows), "audit_runs": len({r["run"] for r in rows}),
        "audit_max": max(abs(r["recon"] - r["lp"]) for r in rows),
        "sums_n": len(sum_err), "sums_max": r4(max(sum_err)),
        "base_depth": base[next(iter(base))]["depth"] / 1e6,
        "b_q": q([r["b"] for r in rows]), "off_q": q([r["a"] - r["lnr"] for r in rows]),
        "frac0_q": q([r["frac0"] for r in rows])}
# old-A (a, b) points for the pseudocount panel: one per pair, mean over seeds
ab_old = collections.defaultdict(list)
for r in rows:
    ab_old[(r["g"], r["kind"], r["src"], r["tgt"])].append((r["lnr"], r["a"], r["b"]))
diag["ab_old"] = [[g, k, r4(v[0][0]), r4(st.mean(x[1] for x in v)), r4(st.mean(x[2] for x in v))]
                  for (g, k, _s, _t), v in ab_old.items()]
# composed through the base (diag_compose.py): arm1 -> base -> arm2 against the direct arm1 -> arm2
P = collections.defaultdict(dict)
for r in tsv(DI / "diag_depth_records_row1.tsv"):
    if r["log2_pred"] == "None" or r["model"] != "real":
        continue
    P[(r["run"], r["track"])][(r["source_pid"].split("__", 1)[1], r["target_pid"].split("__", 1)[1])] = \
        (float(r["log2_pred"]), float(r["log2_true"]))
comp = collections.defaultdict(lambda: ([], []))
for (run, _tr), d in P.items():
    rung, g = run.split("_")[:2]
    for (a, b), (p, t) in d.items():
        if "base__base" in (a, b):
            continue
        pa, ta = d[(a, "base__base")]
        pb, tb = d[("base__base", b)]
        k = (rung, "across" if g == "all" else "per_track")
        comp[k][0].append(abs(p - t))
        comp[k][1].append(abs((pa + pb) - (ta + tb)))
diag["compose"] = [[k[0], k[1], r4(med(v[0])), r4(med(v[1])), len(v[0])] for k, v in sorted(comp.items())]

# ---------------- t126 readings ----------------
depth = [[r["arm"], r["g_version"], r["group"], r4(r["median_abs_err"]), r4(r["q25_abs_err"]),
          r4(r["q75_abs_err"]), r4(r["slope"]), int(r["n_records"])] for r in tsv(R / "depth_table.tsv")]
seedmed = collections.defaultdict(list)
for r in csv.DictReader(gzip.open(R / "depth_points.tsv.gz", "rt"), delimiter="\t"):
    seedmed[(r["arm"], r["g_version"], r["dgroup"], r["seed"])].append(abs(float(r["err"])))
depth_seed = collections.defaultdict(list)
for (a, g, grp, _s), v in seedmed.items():
    depth_seed[f"{a}|{g}|{grp}"].append(r4(med(v)))
knob = [[r["arm"], r["g_version"], r["group"], r["knob"], r4(r["med_rel_nocov"]), int(r["n_pairs"])]
        for r in tsv(R / "knob_table.tsv")]
fit = [[r["arm"], r["g_version"], r["model"], r4(r["crps_mean"]), r4(r["crps_min"]), r4(r["crps_max"]),
        r4(r["oracle_scaled_mean"]), r4(r["scale_error_mean"]), int(r["n_seeds"])] for r in tsv(R / "fit_table.tsv")]
ab = tsv(R / "ab_table.tsv")
abpts = [[r["arm"], r["g_version"], r["kind"], r4(r["ln_r"]), r4(r["desc_a"]), r4(r["desc_b"])] for r in ab]
# the round trip c -> c' -> c: b(c -> c') * b(c' -> c), per (arm, g, seed, depth level) on trained pairs
bb = {(r["arm"], r["g"], r["seed"], r["arm_src"], r["level_src"], r["arm_tgt"], r["level_tgt"]): float(r["desc_b"])
      for r in ab if r["kind"] == "trained"}
prod = collections.defaultdict(list)
for (arm, g, s, as_, ls, at, lt), b in bb.items():
    if as_ == "base":
        back = bb.get((arm, g, s, at, lt, as_, ls))
        if back is not None:
            gv = "across" if g == "all" else "per_track"
            prod[f"{arm}|{gv}"].append(r4(b * back))
lam = [[r["arm"], r["g_version"], r4(r["val_nll_mean"]), r4(r["val_nll_seed_range"]), float(r["lambda"]),
        r["chosen"] == "True", int(r["n_runs"])] for r in tsv(R / "lambda_table.tsv")]
steps = {r["arm"]: [int(r["n_runs"]), int(r["n_at_cap_4000"])] for r in tsv(R / "steps_table.tsv")}
runs = collections.Counter()
for t in "AD":
    seen = set()
    for r in csv.DictReader(gzip.open(R / f"records_{t}.tsv.gz", "rt"), delimiter="\t"):
        if r["run"] not in seen:
            seen.add(r["run"])
            runs[f"{r['run'].split('_')[0]}|{r['model']}|{r['g'] == 'all'}|{r['seed']}"] += 1
runs_n = {"A": sum(v for k, v in runs.items() if k.split("|")[0] in ("A", "AED", "AED1")),
          "D": sum(v for k, v in runs.items() if k.split("|")[0] not in ("A", "AED", "AED1")),
          "seeds": len({k.split("|")[3] for k in runs}),
          "per_track": len({r["g"] for r in ab if r["g"] != "all"})}

# depth levels: trained and held-out targets of base -> depth pairs (depth_points), base depth (count sums)
lv = collections.defaultdict(set)
down = collections.defaultdict(list)
for r in csv.DictReader(gzip.open(R / "depth_points.tsv.gz", "rt"), delimiter="\t"):
    if r["arm_src"] == "base" and r["dgroup"] in ("trained", "heldout"):
        lv[r["dgroup"]].add(r["level_tgt"])
    if r["dgroup"] == "trained" and float(r["log2_true"]) < 0 and r["arm"] in ("AED", "AED1", "A"):
        down[r["arm"]].append(float(r["log2_pred"]))
mb = lambda s: float(s.rstrip("M"))
levels = {"base": diag["base_depth"], "trained": sorted(map(mb, lv["trained"]), reverse=True),
          "heldout": sorted(map(mb, lv["heldout"]), reverse=True)}
# fewer reads in the target: the share of predictions that go UP anyway, and their median log2 scale
down = {a: [r4(sum(v > 0 for v in x) / len(x)), r4(med(x)), len(x)] for a, x in down.items()}
cap = int(next(c for c in open(R / "steps_table.tsv").readline().split() if c.startswith("n_at_cap_")).split("_")[-1])

# the real H3K27ac snippet of the t118 deck (base, 15 M, 3.75 M arms) and its track's true 15 M ratio
snip = json.load(open(HERE.parent / "t118-grid" / "snippet.json"))
locus = snip["locus"]
snip = {k: snip[k] for k in ["base__base/counts25", "depth__15M/counts25"]}
ssum = {s["pid"]: s["score_sum"] for s in sums}
snip_ratio = ssum["C19M16__depth__15M"] / ssum["C19M16__base__base"]

data = {"snip": snip, "locus": locus, "snip_ratio": snip_ratio, "levels": levels, "down": down, "cap": cap,
        "mot_depth": mot_depth, "mot_knob": mot_knob, "diag": diag, "depth": depth,
        "depth_seed": dict(depth_seed), "knob": knob, "fit": fit, "ab": abpts, "prod": dict(prod),
        "lam": lam, "lam_chosen": json.load(open(R / "lambda_chosen.json")), "steps": steps, "runs": runs_n}
src = open(HERE / "deck_src.html").read()
assert src.count("__DATA__") == 1
open(HERE / "index.html", "w").write(src.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print(f"wrote {HERE / 'index.html'}: {len(depth)} depth cells, {len(knob)} knob cells, {len(fit)} fit cells, "
      f"{len(abpts)} (a, b) points, runs A {runs_n['A']} D {runs_n['D']}")
