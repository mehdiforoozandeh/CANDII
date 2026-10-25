"""Rebuild index.html from deck_src.html by embedding the numbers as DATA.

Every number on a slide is read here from a results file, never typed by hand.

Inputs (all gitignored, local):
  cruxvault/results/t125/part1/summary_data.json   t125 part-1 aggregate (cells, readings, depth law,
                                                   explosions; old-grid cells and depth-law values)
  cruxvault/results/t125/part1/run_device.tsv      training device of every part-1 run
  cruxvault/results/t125/part1/depth_law.tsv       per-pair depth ratios (only the snippet's ratio)
  <main checkout>/cruxvault/results/row2/GRID_SUMMARY.md   the old grid's -log10 p explosion tables
  ../t118-grid/snippet.json                        the real H3K27ac snippet used by the t118 deck
  tools/t118/covariates.tsv                        the assay of the chain track
  cruxvault/results/t125/part2/summary_data_h.json t125 part-2 aggregate, 6 of 8 chain designs (job 24966577,
                                                   kit K14, 2026-10-25; the 5-of-8 file is kept in part2_5of8/).
                                                   h alone and in the chain; readings repair, vs_oldgrid,
                                                   push_vs_noise; explosions; merged mixture; fit caps; capped
                                                   NB rows. Whatever designs it holds are drawn; B and B2 were
                                                   stopped by the PI (2026-10-19) and are not rerun.

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

# ---- part 2: the map h (part2/summary_data_h.json, 6 of 8 chains). Every design the file holds is passed on;
#      the missing ones (B, B2, stopped by the PI) are listed from the file, not typed. ----
H = json.load(open(VAULT / "results" / "t125" / "part2" / "summary_data_h.json"))
fid = lambda c: "|".join(str(c.get(k)) for k in ("variant", "window", "rung", "noise", "source"))
H_MET = ["crps_log_all", "crps_log_oracle_scaled_all", "crps_log_scale_error_all", "mse_log1p_all",
         "crps_all", "crps_oracle_scaled_all", "scale_error_all"]
# cells: "class|version id|metric" (alone) or "f design|version id|metric" (chain) -> [mean, wobble, n]
h_alone = {f"{c['mark_class']}|{fid(c)}|{c['metric']}": [r4(c["mean"]), r4(c["seed_wobble"]), c["n"]]
           for c in H["alone_cells"] if c["metric"] in H_MET}
h_chain = {f"{c['f_rung']}|{fid(c)}|{c['metric']}": [r4(c["mean"]), r4(c["seed_wobble"]), c["n"]]
           for c in H["chain_cells"] if c["metric"] in H_MET}
h_rd = []
for x in H["readings"]:
    k, m = x["reading"], x["metric"]
    base = {"k": k, "sc": x["scope"], "f": x.get("f_rung"), "c": x["mark_class"], "m": m,
            "id": fid({**x, "window": {"h_formula_bin": "bin", "h_formula_frag": "frag"}.get(x.get("h_name"))})}
    if k == "repair" and m in ("crps_log_all", "crps_log_oracle_scaled_all", "crps_log_scale_error_all", "crps_all"):
        base.update(t=r4(x["true_mean"]), tw=r4(x["true_wobble"]), o=r4(x["fnb_mean"]), d=r4(x["delta"]),
                    bar=r4(x["bar"]), met=bool(x["abs_delta_exceeds_bar"]),
                    n=x["components"]["pairs"]["n_used"])
    elif k == "vs_oldgrid" and m in ("crps_all", "crps_oracle_scaled_all", "scale_error_all"):
        base.update(h=r4(x["h_mean"]), hw=r4(x["h_wobble"]), o0=r4(x["old_seed0"]),
                    o=r4(x["old_mean"]), ow=r4(x["old_wobble"]), d=r4(x["delta"]), d0=r4(x["delta_seed0"]),
                    bar=r4(x["bar"]), met=bool(x["gain_exceeds_bar"]), met0=bool(x["gain_seed0_exceeds_bar"]),
                    hx=x["h_n_crps_gt_20"], ox=x["old_n_crps_gt_20"], oxn=x["old_n_crps_nonfinite"],
                    n=x["n_pairs_used"])
    elif k == "push_vs_noise" and m in ("crps_log_all", "mse_log1p_all"):
        base.update(p=r4(x["push_mean"]), pw=r4(x["push_wobble"]), q=r4(x["noise_mean"]), qw=r4(x["noise_wobble"]),
                    g=r4(x["gain"]), bar=r4(x["bar"]), met=bool(x["gain_exceeds_bar"]))
    else:
        continue
    h_rd.append(base)
# explosions of the own-noise point CRPS, by scope, with the share from the affine rung-5 versions
h_expl = collections.defaultdict(lambda: {"rec": 0, "gt": 0, "nf": 0, "lnf": 0, "gt_aff5": 0})
for e in H["explosions"]:
    s = h_expl[f"{e['scope']}|{e['kind']}"]
    s["rec"] += e["n_records"]; s["gt"] += e["n_gt_20"]; s["nf"] += e["n_native_nonfinite"]
    s["lnf"] += e["n_log_nonfinite"]
    if e["variant"] in ("affine", "formula_affine") and e["rung"] == 5:
        s["gt_aff5"] += e["n_gt_20"]
old_cell_n = {f"{e['rung']}|{e['g_version']}|{e['mark_class']}": e["n_records"] for e in H["oldgrid_explosions"]}
# the chain track's assay, from the t118 covariates table (column "track", "assay")
COV = VAULT.parent / "tools" / "t118" / "covariates.tsv"
assay = next(r["assay"] for r in csv.DictReader(open(COV), delimiter="\t") if r["track"] == H["chain_track"])
h2 = {"created": H["created_utc"], "track": H["chain_track"], "assay": assay,
      "seeds_true": max(len(c["seeds"]) for c in H["alone_cells"] if c["source"] == "true"),
      "seeds_fnb": max(len(c["seeds"]) for c in H["chain_cells"] if c["source"] == "fnb"), "explode_at": H["explode_at"],
      "alone": [H["alone_present"], H["alone_expected"]],
      "chain": [H["chain_f_present"], H["chain_f_expected"]],
      "f_done": sorted(r.split("_")[0] for r in H["chain_f_finished_runs"]),
      "f_missing": sorted(r.split("_")[0] for r in H["chain_f_runs_missing"]),
      "f_g": sorted({c["f_g_version"] for c in H["chain_cells"]}),
      "cells_alone": h_alone, "cells_chain": h_chain, "rd": h_rd,
      "expl": dict(h_expl), "old_cell_n": old_cell_n,
      # merged mixture per chain f: [design, G, largest per-bin error, largest averaged error, bar, h at or above bar,
      #                              own-noise h checked]
      "merge": [[m["f_run"].split("_")[0], m["mix_groups"], m["rel_err_max"], m["avg_rel_err_max"], m["bar"],
                 m["n_h_rel_err_max_ge_bar"], m["n_h_checked"]] for m in H["merge_errors"]],
      "fits": {t["source"]: t for t in H["fits"]["totals"]},
      "capped": [[c["f_run"].split("_")[0], c["n_pairs"], c["n_pairs_capped"], c["n_capped_rows"]] for c in H["capped"]],
      "notes": H["notes"]}

data = {"snip": snip, "locus": locus, "law_ratio": law_ratio, "cells": cells, "rd": rd, "depth": depth,
        "expl": {"kinds": dict(kinds), "nonzero": nonzero, "old_counts": old_counts_expl,
                 "at": S["explode_at"]},
        "oldp": oldp, "dev": {k: dict(v) for k, v in dev.items()}, "runs": runs,
        "created": S["created_utc"], "h2": h2}
src = open(HERE / "deck_src.html").read()
assert src.count("__DATA__") == 1
open(HERE / "index.html", "w").write(src.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print(f"wrote {HERE / 'index.html'}: {len(cells)} cells, {len(rd)} comparisons, {len(depth)} depth cells; "
      f"part 2: {len(h_alone)} alone cells, {len(h_chain)} chain cells, {len(h_rd)} readings, "
      f"{h2['chain'][0]} of {h2['chain'][1]} chains")
