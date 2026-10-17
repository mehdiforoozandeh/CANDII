"""Rebuild index.html from deck_src.html by embedding the numbers as DATA.

Every number on a slide is read here from a results file, never typed by hand.

Inputs (all gitignored, local):
  cruxvault/results/t125/part1/summary_data.json   t125 part-1 aggregate (cells, readings, depth law,
                                                   explosions; old-grid cells and depth-law values)
  cruxvault/results/t125/part1/run_device.tsv      training device of every part-1 run
  cruxvault/results/t125/part1/depth_law.tsv       per-pair depth ratios (only the snippet's ratio)
  <main checkout>/cruxvault/results/row2/GRID_SUMMARY.md   the old grid's -log10 p explosion tables
  ../t118-grid/snippet.json                        the real H3K27ac snippet used by the t118 deck

    python cruxvault/presentations/t125-eta-h/build_deck.py
"""
import collections, csv, json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
VAULT = HERE.parent.parent
R1 = VAULT / "results" / "t125" / "part1"
# results/ is gitignored, so the old grid lives only in the main checkout, not in a worktree
OLD = next(p for p in [VAULT / "results" / "row2",
                       pathlib.Path.home() / "Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results/row2"]
           if (p / "GRID_SUMMARY.md").exists())

r4 = lambda v: None if v is None else round(float(v), 5)

# ---- the real snippet (t118 deck): base, 15 M and 3.75 M arms, counts ----
snip = json.load(open(HERE.parent / "t118-grid" / "snippet.json"))
locus = snip["locus"]
snip = {k: snip[k] for k in ["base__base/counts25", "depth__15M/counts25", "depth__3.75M/counts25"]}
law_ratio = next(float(r["depth_log2_ratio_true"])
                 for r in csv.DictReader(open(R1 / "depth_law.tsv"), delimiter="\t")
                 if r["source_pid"] == "C19M16__base__base" and r["target_pid"] == "C19M16__depth__3.75M")

S = json.load(open(R1 / "summary_data.json"))
assert S["runs_present"] == S["n_expected_runs"] and not S["runs_missing"]

# ---- cells: "src|rung|g|class|metric" -> [mean, wobble, oracle-scaled, wob, scale error, wob] ----
SRC = {"t125 real": "real", "t125 nocov": "twin", "old grid": "old"}
cells = {}
for c in S["cells"]:
    v = [r4(c["mean"]), r4(c["seed_wobble"])]
    sp = c.get("split")
    if sp:
        v += [r4(sp["crps_oracle_scaled_all"]["mean"]), r4(sp["crps_oracle_scaled_all"]["seed_wobble"]),
              r4(sp["scale_error_all"]["mean"]), r4(sp["scale_error_all"]["seed_wobble"])]
    cells["|".join([SRC[c["source"]], c["rung"], c["g_version"], c["mark_class"], c["metric"]])] = v

# ---- comparisons: "reading|metric|rung|g|class" -> [gain, bar, met, this, wob, other, wob,
#      oracle-scaled gain, bar, scale-error gain, bar] ----
rd = {}
for x in S["readings"]:
    if x["reading"] not in ("vs_oldgrid", "beatstwin", "vs_oldspread"):
        continue
    c = x["components"]
    v = [r4(x["value"]), r4(x["bar"]), bool(x["met"]), r4(c["this"]), r4(c["wobble_this"]),
         r4(c["other"]), r4(c["wobble_other"])]
    sg = c.get("split_gain")
    if sg:
        v += [r4(sg["crps_oracle_scaled_all"]["gain"]), r4(sg["crps_oracle_scaled_all"]["bar"]),
              r4(sg["scale_error_all"]["gain"]), r4(sg["scale_error_all"]["bar"])]
    rd["|".join([x["reading"], x["metric"], x["rung"], x["g_version"], x["mark_class"]])] = v

# ---- depth law: [rung, g, class, value, wobble, seed min, seed max, within bar, twin, twin wobble,
#      old grid, old wobble, gain vs old, bar vs old, gain clears, bar, n pairs] ----
depth = []
for x in S["depth_law"]:
    c = x["components"]
    depth.append([x["rung"], x["g_version"], x["mark_class"], r4(x["value"]), r4(x["seed_wobble"]),
                  r4(min(c["per_seed"])), r4(max(c["per_seed"])), bool(x["met"]),
                  r4(c["nocov_value"]), r4(c["nocov_seed_wobble"]),
                  r4(c.get("oldgrid_value")), r4(c.get("oldgrid_seed_wobble")),
                  r4(c.get("gain_vs_oldgrid")), r4(c.get("bar_vs_oldgrid")),
                  bool(c.get("gain_vs_oldgrid_exceeds_bar")), r4(x["bar"]), c["n_pairs"]])

# ---- explosions (CRPS > 20) in part 1, by kind, and every nonzero cell ----
kinds = collections.defaultdict(lambda: [0, 0, 0])
nonzero = []
for e in S["explosions"]:
    k = kinds[f"{e['kind']}/{e['eval']}"]
    k[0] += e["n_records"]; k[1] += e["n_crps_gt_20"]; k[2] += e["n_crps_nonfinite"]
    if e["n_crps_gt_20"] or e["n_crps_nonfinite"]:
        nonzero.append([e["rung"], e["g_version"], e["mark_class"], e["model"], f"{e['kind']}/{e['eval']}",
                        e["n_records"], e["n_crps_gt_20"], e["n_crps_nonfinite"]])
old_counts_expl = [sum(e["n_records"] for e in S["oldgrid_explosions"]),
                   sum(e["n_crps_gt_20"] for e in S["oldgrid_explosions"])]

# ---- the old grid's -log10 p explosions: GRID_SUMMARY.md tables, summed over classes and both g ----
oldp = {"row 1": [0] * 4, "row 2": [0] * 4, "twin": [0] * 4}
sec = None
in_expl = False
for line in open(OLD / "GRID_SUMMARY.md"):
    if line.startswith("### "):
        sec = line[4:].strip(); in_expl = False
    elif line.startswith("Exploding"):
        in_expl = True
    elif in_expl and sec and sec.endswith("-log10 p") and line.startswith("| ") and not line.startswith("| class"):
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if cols[1] in oldp:
            for i in range(4):
                oldp[cols[1]][i] += int(cols[2 + i])
assert sum(oldp["row 1"]) and sum(oldp["row 2"]), "old-grid p explosion tables not found"

# ---- runs and training device per design (run_device.tsv) ----
dev = collections.defaultdict(lambda: collections.Counter())
gver, models, seeds = set(), set(), set()
for r in csv.DictReader(open(R1 / "run_device.tsv"), delimiter="\t"):
    rung, g, _sp, model, seed = r["run_name"].split("_")
    dev[rung][r["train_device"]] += 1
    gver.add(g); models.add(model); seeds.add(seed)
runs = {"n": sum(sum(v.values()) for v in dev.values()), "gver": len(gver), "models": len(models),
        "seeds": len(seeds), "per_track": len(gver - {"all"})}

data = {"snip": snip, "locus": locus, "law_ratio": law_ratio, "cells": cells, "rd": rd, "depth": depth,
        "expl": {"kinds": dict(kinds), "nonzero": nonzero, "old_counts": old_counts_expl,
                 "at": S["explode_at"]},
        "oldp": oldp, "dev": {k: dict(v) for k, v in dev.items()}, "runs": runs,
        "created": S["created_utc"]}
src = open(HERE / "deck_src.html").read()
assert src.count("__DATA__") == 1
open(HERE / "index.html", "w").write(src.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print(f"wrote {HERE / 'index.html'}: {len(cells)} cells, {len(rd)} comparisons, {len(depth)} depth cells")
