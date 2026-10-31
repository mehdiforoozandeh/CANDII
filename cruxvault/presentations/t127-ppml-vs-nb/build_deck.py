"""Rebuild index.html from deck_src.html by embedding the numbers as DATA.

Every number on a slide is read here from a results file, never typed by hand.

Inputs (all gitignored, local):
  cruxvault/results/t127/deltas.csv               nb vs ppml per rung: means, Δ, seed wobble (trained pairs)
  cruxvault/results/t127/depth_law.csv            depth law per rung × loss × pair kind (trained / never-trained)
  cruxvault/results/t127/depth_law_by_class.csv   the worst-pair check per mark class
  cruxvault/results/t127/records.jsonl.gz         per-pair records: a re-pooled CRPS split, DNase totals,
                                                  training stop steps (pooled here by the t118 convention)
  cruxvault/results/t127/extra/*.json             per-pair observed totals Σy: does the data obey the law?
  cruxvault/results/t127/row1_sanity_pairs.json   the nb arm against the earlier row-1 runs
  cruxvault/results/t127/tasks.tsv                the run grid (rungs × losses × tracks × seeds)
  cruxvault/results/h19/depth_pairs/out_raw.txt, out_raw_row2.txt   motivation: trained vs never-trained,
                                                  every design of rows 1 and 2
  ../t118-grid/snippet.json                       the real H3K27ac snippet used by the t118 deck

Pooling (t118 convention, results/t127/analyse.py): mean over pairs per track, macro mean over the
7 tracks per seed, mean over seeds; seed wobble = largest pairwise |gap| of the 3 per-seed values;
the wobble beside a Δ is the larger of the two arms' wobbles.

    /Users/mforooz/miniforge3/envs/candii/bin/python cruxvault/presentations/t127-ppml-vs-nb/build_deck.py
"""
import csv, glob, gzip, itertools, json, math, pathlib, statistics

HERE = pathlib.Path(__file__).resolve().parent
VAULT = HERE.parent.parent
R = VAULT / "results" / "t127"
H19 = VAULT / "results" / "h19" / "depth_pairs"
RUNGS, LOSSES, SEEDS = ("A", "A0", "D"), ("nb", "ppml"), (0, 1, 2)

r5 = lambda v: None if v is None or v == "" else round(float(v), 5)
wob = lambda v: max(abs(a - b) for a, b in itertools.combinations(v, 2))

# ---- deltas.csv: "aspect|rung" -> [nb, ppml, delta, wob] ----
deltas = {f"{r['aspect']}|{r['rung']}": [r5(r["nb"]), r5(r["ppml"]), r5(r["delta"]), r5(r["wob"])]
          for r in csv.DictReader(open(R / "deltas.csv"))}

# ---- depth_law.csv: "rung|loss|trained|law" -> [median |log2 err|, wobble, worst check, wobble, n pairs] ----
depth = {f"{r['rung']}|{r['loss']}|{r['depth_source']}":
         [r5(r["median_abs_log2_err"]), r5(r["wob_median"]), r5(r["worst_check_value"]), r5(r["wob_worst"]),
          int(r["n_pairs"])] for r in csv.DictReader(open(R / "depth_law.csv"))}
depth_mc = {f"{r['rung']}|{r['loss']}|{r['depth_source']}|{r['mark_class']}": [r5(r["worst_check_value"]),
            r5(r["wob_worst"])] for r in csv.DictReader(open(R / "depth_law_by_class.csv"))}

# ---- per-pair records ----
recs = [json.loads(l) for l in gzip.open(R / "records.jsonl.gz", "rt")]


def pool(rows, metric):
    """t118 pooling -> (mean over seeds, seed wobble, per-seed values)."""
    per = {}
    for s in SEEDS:
        tr = {}
        for x in rows:
            if x["run_seed"] == s:
                tr.setdefault(x["track"], []).append(x[metric])
        per[s] = statistics.fmean(statistics.fmean(v) for v in tr.values())
    v = [per[s] for s in SEEDS]
    return statistics.fmean(v), wob(v), v


# trained-pair CRPS and its oracle-scaled / scale-error split, re-pooled from the raw records: the
# deck draws deltas.csv, and check_deck compares the two. Never-trained records carry no split, so
# the deck shows no never-trained CRPS at all.
split = {}
for rung in RUNGS:
    for loss in LOSSES:
        rows = [x for x in recs if x["kind"] == "trained" and x["eval"] == "score" and x["run_rung"] == rung
                and x["run_loss"] == loss]
        split[f"{rung}|{loss}"] = {m: [r5(a) for a in pool(rows, m)[:2]] for m in
                                   ("crps_all", "crps_oracle_scaled_all", "scale_error_all")}

# DNase totals: Σμ / Σy on the trained pair "duplicates kept -> base", per rung, loss and seed
dn_pair = "dedup:off->base:base"
dnase = {}
for x in recs:
    if x["kind"] == "trained" and x["eval"] == "score" and x["mark_class"] == "DNase" and x["knob_combo"] == dn_pair:
        dnase.setdefault(f"{x['run_rung']}|{x['run_loss']}", {})[x["run_seed"]] = r5(x["mu_sum_over_y_sum"])
dnase = {k: [v[s] for s in SEEDS] for k, v in dnase.items()}
dn_assay = next(x["assay"] for x in recs if x["mark_class"] == "DNase")

# training: matching stop steps between the paired nb and ppml runs (same rung, track, seed)
best = {(x["run_rung"], x["run_g"], x["run_seed"], x["run_loss"]): x["run_best_step"] for x in recs}
pairs = {k[:3] for k in best}
same_stop = [sum(best[p + ("nb",)] == best[p + ("ppml",)] for p in pairs), len(pairs)]
sanity = json.load(open(R / "row1_sanity_pairs.json"))

# ---- do the observed totals obey the law? Σy of each depth arm, all ordered arm -> arm pairs ----
tot = {}
for f in glob.glob(str(R / "extra" / "*.json")):
    for x in json.load(open(f))["records"]:
        if x["eval"] == "score":
            tot[x["target_pid"]] = x["sum_y"]
dl_true = {(x["source_pid"], x["target_pid"]): x["depth_log2_ratio_true"] for x in recs if x["kind"] == "depthlaw"
           and x["depth_source"] == "law"}
law_err = [abs(math.log2(tot[t] / tot[s]) - v) for (s, t), v in dl_true.items()]
target_law = {"median": r5(statistics.median(law_err)), "max": r5(max(law_err)), "n": len(law_err)}

# ---- the run grid (tasks.tsv) ----
tasks = list(csv.DictReader(open(R / "tasks.tsv"), delimiter="\t"))
grid = {"n": len(tasks), "tracks": len({t["g"] for t in tasks}), "seeds": len({t["seed"] for t in tasks}),
        "rungs": sorted({t["rung"] for t in tasks}), "losses": sorted({t["loss"] for t in tasks}),
        "space": sorted({t["space"] for t in tasks}), "model": sorted({t["model"] for t in tasks})}
grid["cell"] = {f"{r}|{l}": sum(t["rung"] == r and t["loss"] == l for t in tasks) for r in RUNGS for l in LOSSES}
classes = {}
for x in recs:
    classes.setdefault(x["mark_class"], set()).add(x["track"])
classes = {k: len(v) for k, v in classes.items()}

# ---- motivation (h19): real model, one g per track, median |log2 err|, trained vs never-trained ----
motiv = {}
for fn in ("out_raw.txt", "out_raw_row2.txt"):
    for line in open(H19 / fn):
        c = line.split()
        if len(c) == 11 and c[0] == "real" and c[2] == "per_track":
            motiv.setdefault(c[1], {})[c[3]] = r5(c[8])
assert len(motiv) == 8 and all(len(v) == 2 for v in motiv.values()), motiv

# ---- the real snippet: base counts, for the loss-weight schematic ----
snip = json.load(open(HERE.parent / "t118-grid" / "snippet.json"))
snippet = {"locus": snip["locus"], "base": snip["base__base/counts25"]}

data = {"deltas": deltas, "depth": depth, "depth_mc": depth_mc, "split": split, "dnase": dnase,
        "dn_pair": dn_pair, "dn_assay": dn_assay, "same_stop": same_stop,
        "sanity": {"median": r5(sanity["median_abs_rel_diff"]), "n": sanity["n"]},
        "target_law": target_law, "grid": grid, "classes": classes, "motiv": motiv, "snip": snippet}
src = open(HERE / "deck_src.html").read()
assert src.count("__DATA__") == 1
open(HERE / "index.html", "w").write(src.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print(f"wrote {HERE / 'index.html'}: {len(deltas)} delta rows, {len(depth)} depth cells, {grid['n']} runs, "
      f"law target median {target_law['median']} over {target_law['n']} pairs")
