"""Rebuild index.html from deck_src.html: the unified phase-1 + phase-2 deck of the symmetric
encoder-decoder (the "pivot translator"), for a lab audience that does not know the design.

Every number on a slide is read here from a results file, never typed by hand (the pattern of
../t128-encdec-sym/ and ../t128p2-encdec/). Rebuild after new readings land:

    bash tools/t128/readings/run_all.sh          # phase 1 tables
    bash tools/t128p2/readings/run_all.sh        # phase 2 tables
    /Users/mforooz/miniforge3/envs/candii/bin/python cruxvault/presentations/t128-pivot/build_deck.py

Inputs (gitignored; the main checkout's copies by default, override with T128_RESULTS / T128P2_RESULTS):
  phase 1  results/t128/   n_table, knob_table, twin_table, swap_table, collapse_table, depth_table,
                           cclean_table, lambda_table, steps_table, runs_fir/runs_nibi
  phase 2  results/t128p2/ n_table, m1_never_table, m3_trained_table, m4_heldout_table,
                           m5_depthlaw_table, m6_hop1_table, m7_agree_table, m8_swap_table,
                           m9_steps_table, runs_fir, cclean_table

NAMES maps every internal arm code to the plain name the deck uses; the build fails if a code
leaks into DATA. The sources manifest records the sha256 of every file read. Stdlib only.
"""
import collections, csv, gzip, hashlib, json, math, os, pathlib, re, statistics as st

HERE = pathlib.Path(__file__).resolve().parent
MAIN = pathlib.Path("/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results")
R1 = pathlib.Path(os.environ.get("T128_RESULTS", MAIN / "t128"))
R2 = pathlib.Path(os.environ.get("T128P2_RESULTS", MAIN / "t128p2"))

# internal code -> (phase, plain name). The plain names are the deck's vocabulary (slide "the variants").
P1N = {"D": "direct translator", "Dviabase": "two-step via the base",
       "DS": "learned pivot", "DSrt": "learned pivot, round-trip loss only",
       "DSrtvae": "learned pivot, round-trip loss only, noisy middle",
       "DSvae": "learned pivot, noisy middle", "DSbox": "boxed pivot",
       "DSl0.1": "learned pivot, agreement weight 0.1", "DSl1": "learned pivot, agreement weight 1",
       "DSl10": "learned pivot, agreement weight 10"}
P2N = {"D": "direct translator", "Dviabase": "two-step via the base", "DSfree": "learned pivot",
       "DSfixed": "base-labelled pivot", "DSanchor": "anchored pivot"}
GN = {"per_track": "one g per track", "across": "one g across tracks"}
CODES = re.compile(r"\b(DSfree|DSfixed|DSanchor|Dviabase|DSrtvae|DSrt|DSvae|DSbox|DSl0\.1|DSl10|DSl1|DS|old D)\b")

read = []


def tsv(root, name):
    p = root / name
    read.append(p)
    op = gzip.open(p, "rt") if name.endswith(".gz") else open(p)
    with op as f:
        return list(csv.DictReader(f, delimiter="\t"))


def num(v, nd=5):
    if v is None or v == "" or str(v).lower() in ("nan", "none", "<na>"):
        return None
    f = float(v)
    return None if math.isnan(f) else round(f, nd)


def seeds(r, stems):
    out = [num(r.get(s)) for s in stems]
    return [v for v in out if v is not None]


S012 = ("seed0", "seed1", "seed2")
Ss = lambda stem: tuple(f"{stem}_s{i}" for i in (0, 1, 2))


def crow(r, stem_seeds=S012, pre=""):
    """{v, os, se, sp, seeds}: a CRPS always travels with its split and its seed spread."""
    return {"v": num(r[f"{pre}crps_all"]), "os": num(r[f"{pre}crps_oracle_scaled_all"]),
            "se": num(r[f"{pre}scale_error_all"]), "sp": num(r["seed_spread"]), "seeds": seeds(r, stem_seeds)}


# ================= phase 1 (4000-step cap) =================
p1n = collections.defaultdict(lambda: {"assigned": 0, "done": 0})
for r in tsv(R1, "n_table.tsv"):
    if r["model"] in ("real", "nocov") and r["arm"] in P1N and r["cluster"] not in ("fir", "nibi"):  # per-cluster rows repeat "both"
        k = P1N[r["arm"]] + (" (no-settings twin)" if r["model"] == "nocov" else "")
        p1n[k]["assigned"] += int(float(r["assigned"]))
        p1n[k]["done"] += int(float(r["LAW_DONE"] or 0))
p1_trained_runs = {k: v for k, v in p1n.items() if not k.startswith(("direct", "two-step"))}

p1never = {}        # plain name -> g -> {v, os, se, sp, seeds, n, label_vs_default, label_vs_direct}
for r in tsv(R1, "knob_table.tsv"):
    if r["group"] == "never" and r["knob"] == "all" and r["arm"] in P1N:
        d = crow(r)
        d.update(n=int(float(r["n_runs"])), vs_default=r["label_vs_DS"], gap_default=num(r["gap_vs_DS"]),
                 vs_direct=r["label_vs_D"], gap_direct=num(r["gap_vs_D"]))
        p1never.setdefault(P1N[r["arm"]], {})[GN[r["g_version"]]] = d

p1twin = {}         # gain over the no-settings twin, trained and never-trained pairs
for r in tsv(R1, "twin_table.tsv"):
    if r["knob"] == "all" and r["group"] in ("trained", "never") and r["arm"] in P1N:
        p1twin.setdefault(P1N[r["arm"]], {}).setdefault(GN[r["g_version"]], {})[r["group"]] = {
            "v": num(r["gain_median"]), "sp": num(r["seed_spread"]), "seeds": seeds(r, S012),
            "vs_trained": r["label_vs_trained"]}

p1swap = {}
for r in tsv(R1, "swap_table.tsv"):
    if r["knob"] == "all" and r["model"] == "real" and r["arm"] in P1N:
        p1swap.setdefault(P1N[r["arm"]], {})[GN[r["g_version"]]] = {
            "swap": num(r["swap_median_abs_log_ratio"]), "swap_sp": num(r["seed_spread"]),
            "ident": num(r["identity_median_abs_log_ratio"]), "ident_sp": num(r["ident_seed_spread"])}

p1collapse = {}
for r in tsv(R1, "collapse_table.tsv"):
    if r["knob"] == "all" and r["model"] == "real" and r["arm"] in P1N:
        p1collapse.setdefault(P1N[r["arm"]], {})[GN[r["g_version"]]] = {
            "corr": num(r["corr_z_x"]), "absdiff": num(r["median_abs_z_minus_x"]), "sp": num(r["seed_spread"])}

p1depth = {}
for r in tsv(R1, "depth_table.tsv"):
    if r["arm"] in P1N:
        p1depth.setdefault(P1N[r["arm"]], {}).setdefault(GN[r["g_version"]], {})[r["group"]] = {
            "v": num(r["median_abs_err"]), "sp": num(r["seed_spread"])}

p1steps = {}
for r in tsv(R1, "steps_table.tsv"):
    if r["model"] == "real" and r["arm"] in P1N:
        p1steps[P1N[r["arm"]]] = {"n": int(float(r["n_runs"])), "at_cap": int(float(r["n_at_cap_4000"])),
                                  "best_median": num(r["best_step_median"])}

p1drift = collections.defaultdict(list)    # how far the learned middle setting moved from its start
for r in tsv(R1, "cclean_table.tsv"):
    if r["model"] == "real" and r["arm"] == "DS" and num(r["drift_from_init"]) is not None:
        p1drift[GN[r["g_version"]]].append(abs(num(r["drift_from_init"])))
p1drift = {g: round(st.median(v), 4) for g, v in p1drift.items()}

p1lambda = {}
for r in tsv(R1, "lambda_table.tsv"):
    if r["chosen"] == "True":
        p1lambda[GN[r["g_version"]]] = num(r["lam"])

# ================= phase 2 (12,000-step cap) =================
p2n = {P2N[r["arm"]]: {"assigned": int(float(r["assigned"])),
                       "done": int(float(r["LAW_DONE"] or r["viabase_LAW_DONE"] or 0))}
       for r in tsv(R2, "n_table.tsv")}

p2never = {}        # plain name -> g -> subgroup -> CRPS with split and labels
for r in tsv(R2, "m1_never_table.tsv"):
    d = crow(r, Ss("crps"))
    d.update(n=int(float(r["n_runs"])), vs_direct=r["label_vs_D"], gap_direct=num(r["gap_vs_D"]),
             vs_via=r["label_vs_Dviabase"], gap_via=num(r["gap_vs_Dviabase"]))
    p2never.setdefault(P2N[r["arm"]], {}).setdefault(GN[r["g_version"]], {}).setdefault(r["knob"], {})[r["subgroup"]] = d

p2trained = {}
for r in tsv(R2, "m3_trained_table.tsv"):
    if r["knob"] == "all":
        d = crow(r, Ss("crps"))
        d.update(vs_direct=r["label_vs_D"], gap_direct=num(r["gap_vs_D"]))
        p2trained.setdefault(P2N[r["arm"]], {})[GN[r["g_version"]]] = d

p2heldout = {}
for r in tsv(R2, "m4_heldout_table.tsv"):
    if r["direction"] == "both":
        d = crow(r, Ss("crps"))
        d.update(vs_direct=r["label_vs_D"], gap_direct=num(r["gap_vs_D"]))
        p2heldout.setdefault(P2N[r["arm"]], {})[GN[r["g_version"]]] = d

p2depth = {}
for r in tsv(R2, "m5_depthlaw_table.tsv"):
    p2depth.setdefault(P2N[r["arm"]], {}).setdefault(GN[r["g_version"]], {})[r["group"]] = {
        "v": num(r["median_abs_err"]), "sp": num(r["seed_spread"]), "seeds": seeds(r, Ss("median_abs_err")),
        "vs_direct": r["label_vs_D"], "vs_via": r["label_vs_Dviabase"]}

HOPSRC = {"decoder arm->base (trained record)": "decoder", "encoder NB head (hop1.json)": "encoder"}
p2hop = {}          # hop 1: the predicted base track against the real base, trained arm -> base pairs
for r in tsv(R2, "m6_hop1_table.tsv"):
    if r["knob"] == "all" and r["group"] == "trained":
        d = crow(r, Ss("crps"))
        d.update(vs_direct=r["label_vs_D"], gap_direct=num(r["gap_vs_D"]))
        p2hop.setdefault(P2N[r["arm"]], {}).setdefault(GN[r["g_version"]], {})[HOPSRC[r["source"]]] = d

p2agree = {}        # middle consistency: |Z - Z'| / |x - x'| for two arms of one track (two knobs differ)
for r in tsv(R2, "m7_agree_table.tsv"):
    if r["n_knobs_differ"] == "2":
        p2agree.setdefault(P2N[r["arm"]], {})[GN[r["g_version"]]] = {
            "v": num(r["median_ratio_z_over_x"]), "sp": num(r["median_ratio_z_over_x_seed_spread"]),
            "seeds": seeds(r, Ss("median_ratio_z_over_x"))}

p2swap = {}
for r in tsv(R2, "m8_swap_table.tsv"):
    if r["knob"] == "all":
        p2swap.setdefault(P2N[r["arm"]], {})[GN[r["g_version"]]] = {
            "swap": num(r["swap_median_abs_log_ratio"]), "swap_sp": num(r["seed_spread"]),
            "ident": num(r["identity_median_abs_log_ratio"]), "ident_sp": num(r["ident_seed_spread"])}

p2steps = {}
for r in tsv(R2, "m9_steps_table.tsv"):
    p2steps.setdefault(P2N[r["arm"]], {})[GN[r["g_version"]]] = {
        "n": int(r["n_runs"]), "cap": int(float(r["max_steps"])), "last10": int(float(r["n_best_last10"])),
        "best_median": num(r["best_step_median"]), "run_median": num(r["steps_run_median"])}
p2best = collections.defaultdict(list)     # one dot per run: its best step
for r in tsv(R2, "runs_fir.tsv"):
    if r["TRAIN_DONE"] == "1":
        p2best[P2N[r["arm"]]].append(int(float(r["best_step"])))

data = {"p1n": dict(p1n), "p1_trained_runs": sum(v["done"] for v in p1_trained_runs.values()),
        "p1_assigned": sum(v["assigned"] for v in p1_trained_runs.values()),
        "p1never": p1never, "p1twin": p1twin, "p1swap": p1swap, "p1collapse": p1collapse, "p1depth": p1depth,
        "p1steps": p1steps, "p1drift": p1drift, "p1lambda": p1lambda,
        "p2n": p2n, "p2never": p2never, "p2trained": p2trained, "p2heldout": p2heldout, "p2depth": p2depth,
        "p2hop": p2hop, "p2agree": p2agree, "p2swap": p2swap, "p2steps": p2steps, "p2best": dict(p2best)}
blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
m = CODES.search(blob)
assert not m, "arm code leaked into DATA: " + m.group(0)

read += [HERE / "build_deck.py", HERE / "deck_src.html"]
sources = {}
for p in read:
    p = p.resolve()
    key = ("results/" + p.relative_to(MAIN.resolve()).as_posix()) if MAIN.resolve() in p.parents \
        else "presentations/t128-pivot/" + p.name
    sources[key] = hashlib.sha256(p.read_bytes()).hexdigest()
src = (HERE / "deck_src.html").read_text(encoding="utf-8")
assert src.count("__DATA__") == 1 and src.count("__SOURCES__") == 1
out = src.replace("__SOURCES__", json.dumps(sources, sort_keys=True)).replace("__DATA__", blob)
(HERE / "index.html").write_text(out, encoding="utf-8")
print(f"wrote {HERE / 'index.html'} ({len(out):,} bytes) from {len(sources)} files")
