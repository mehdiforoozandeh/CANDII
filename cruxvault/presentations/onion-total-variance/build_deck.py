"""Rebuild index.html from deck_src.html: the lab deck on how the law of total variance formulates
the onion variance decomposition (PI request 2026-11-13).

Paper arc (prezit references/paper-arc.md). Every number on a slide is read here from a vault file
and embedded as DATA; the page types none. The build writes the deck-sources manifest
(vault-relative path -> sha256 of every file it read) so `crux deck --verify` can prove the deck is
current. No vault question anchors this deck: the study slug is `onion-total-variance`.

Inputs (vault-relative):
  presentations/onion-total-variance/sim_figures.json   SIMULATED illustrations, written by
                                                         simulate_figures.py beside this file
  results/t233/AUDIT.md                                  the audit of the scorer against the identity
  results/t234/AUDIT_TVg_seed0.md                        the same audit after the fixes (seed 0)
  results/t183/DATASET_EIC_V1.md, results/t183/rung_pairs.tsv   the frozen dataset version 1
  results/t185/SUBMITTED.md                              the evaluation set on that dataset
  INSTRUCTIONS/peeling_the_onion/onion total variance.md          explained fraction in simulation
  INSTRUCTIONS/peeling_the_onion/long-horizon notepad.md          second experiments (count of tracks)

`results/` is gitignored: the build reads this worktree's copies. A regex that no longer matches
stops the build and names the file.

    /Users/mforooz/miniforge3/envs/candii/bin/python build_deck.py [VAULT]
Stdlib only.
"""
import csv, hashlib, json, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
VAULT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent.parent
read = []


def text(rel):
    p = VAULT / rel
    read.append(p)
    return p.read_bytes().decode("utf-8")


def grab(rel, src, pattern, n=1):
    m = re.search(pattern, src, re.S)
    if not m:
        sys.exit(f"build_deck: pattern not found in {rel}: {pattern}")
    return [m.group(i + 1) for i in range(n)] if n > 1 else m.group(1)


def num(s):
    return float(s.replace(",", ""))


def audit_rows(rel):
    """Rows of an AUDIT.md table: {place: {planted, est, lo, hi, defect}}."""
    out = {}
    for line in text(rel).splitlines():
        if not line.startswith("| ") or line.startswith("| place") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) < 6:
            continue
        place, _, planted, est, _, defect = cells[:6]
        m = re.match(r"([-+0-9.e]+|nan) \[([-+0-9.e]+|nan), ([-+0-9.e]+|nan)\]", est)
        row = dict(defect=defect == "yes", planted=None, est=None, lo=None, hi=None)
        try:
            row["planted"] = float(planted.split()[0])
        except ValueError:
            pass
        if m:
            row.update(est=float(m.group(1)), lo=float(m.group(2)), hi=float(m.group(3)))
        out[place] = row
    return out


# --- simulated illustrations -------------------------------------------------------------
sim = json.loads(text("presentations/onion-total-variance/sim_figures.json"))

# --- the audit, before and after the fixes -----------------------------------------------
before = audit_rows("results/t233/AUDIT.md")
after = audit_rows("results/t234/AUDIT_TVg_seed0.md")
# plain name of each defect of the first audit -> (row in the first audit, row after the fixes, status)
DEFECTS = [
    ("The layer table stored twice the pair value", "layers.py:93,111 (pair_variance)",
     "layers.py layer_table (pair value)", "fixed"),
    ("The halves rung read one track twice", "layers.py:303-306 with build_tables.py:347-349",
     "layers.py run (a == b) with build_tables.py:347-349", "fixed"),
    ("A layer was taken across two different cell sets", "layers.py:180-183 (increment over another cell set)",
     "layers.py layer_table (increment over another cell set)", "fixed"),
    ("A log1p layer was set against another depth",
     "plan section A (4); layers.py:295 vs score.py:840 (headline 1 at the target depth)",
     "layers.py run (`depth` column; plan section A (4))", "depth now recorded"),
    ("Nested order with crossed donor and experiment", "layers.py:41-42,145-148 (rung order)",
     "layers.py lower_terms (experiment layer, crossed)", "redefined: sibling layers"),
    ("Mixed model nested the donor in the experiment", "mixed_model.py:277-278 (replicate = experiment/donor)",
     "mixed_model.py select_observations + decompose (experiment row, crossed donor)", "fixed"),
    ("Mixed model fitted depth as one slope", "mixed_model.py:5,345-348 (one fixed slope on log2 depth)",
     "mixed_model.py sufficient_stats (thinning to one depth)", "fixed"),
    ("Noise floor on log1p, even halves", "metrics.py:195-226, score.py:843-844 (even halves)",
     "metrics.py noise_floor, score.py (even halves)", "kept; size reported"),
    ("Noise floor on log1p, unequal halves", "metrics.py:195-226, score.py:843-844 (unequal halves 0.3 / 0.7)",
     "metrics.py noise_floor, score.py (unequal halves 0.3 / 0.7)", "partly fixed; size reported"),
    ("Calibration ratio read against the halves only", "metrics.py:481-483, score.py:836 (headline 7, pseudoreplicate)",
     "metrics.py variance_ratio, score.py (headline 7, pseudoreplicate: lower-bound check)", "redefined: two bounds"),
]
defects = []
for name, b, a, status in DEFECTS:
    rb, ra = before[b], after[a]
    if not rb["defect"]:
        sys.exit(f"build_deck: {b} is not a defect in the first audit")
    defects.append(dict(name=name, status=status,
                        b=dict(planted=rb["planted"], est=rb["est"], lo=rb["lo"], hi=rb["hi"]),
                        a=dict(planted=ra["planted"], est=ra["est"], lo=ra["lo"], hi=ra["hi"], defect=ra["defect"])))
n_before_rows, n_before_def = len(before), sum(r["defect"] for r in before.values())
if n_before_def != len(DEFECTS):
    sys.exit(f"build_deck: the first audit has {n_before_def} defects, the deck lists {len(DEFECTS)}")
n_after_rows, n_after_def = len(after), sum(r["defect"] for r in after.values())

H7 = [("halves of the target (half depth)", "metrics.py variance_ratio, score.py (headline 7, pseudoreplicate: lower-bound check)"),
      ("sequencing", "metrics.py variance_ratio via per_rung_variance_ratio (sequencing)"),
      ("technical", "metrics.py variance_ratio via per_rung_variance_ratio (technical)"),
      ("isogenic", "metrics.py variance_ratio via per_rung_variance_ratio (isogenic)"),
      ("anisogenic", "metrics.py variance_ratio via per_rung_variance_ratio (anisogenic)"),
      ("other experiment", "metrics.py variance_ratio via per_rung_variance_ratio (other_experiment)"),
      ("shares only the cell type", "metrics.py variance_ratio (pair that shares only the cell type; upper bound)"),
      ("other experiment, one donor", "score.py H7_UPPER_RUNG (other_experiment, one donor per cell type)"),
      ("other cell type", "metrics.py variance_ratio via per_rung_variance_ratio (other_cell_type)")]
h7 = [dict(lab=l, v=after[k]["est"], lo=after[k]["lo"], hi=after[k]["hi"], planted=after[k]["planted"]) for l, k in H7]

CROSSED = [("donor", "layers.py lower_terms (donor layer, crossed)"),
           ("experiment", "layers.py lower_terms (experiment layer, crossed)"),
           ("cell type", "layers.py lower_terms (cell type layer, crossed)")]
crossed = [dict(lab=l, v=after[k]["est"], lo=after[k]["lo"], hi=after[k]["hi"], planted=after[k]["planted"])
           for l, k in CROSSED]
nested_exp = before["layers.py:41-42,145-148 (rung order)"]
ident = dict(total=before["total_variance.py:simulate"]["planted"],
             est=before["total_variance.py:simulate"]["est"], lo=before["total_variance.py:simulate"]["lo"],
             hi=before["total_variance.py:simulate"]["hi"])
budget_txt = text("results/t234/AUDIT_TVg_seed0.md")
budget_max = num(grab("results/t234/AUDIT_TVg_seed0.md", budget_txt, r"max /lhs - rhs/ = ([0-9.e+-]+)"))

# --- explained fraction per layer, in simulation (the formal note) -----------------------
NOTE = "INSTRUCTIONS/peeling_the_onion/onion total variance.md"
note = text(NOTE)
ef1 = grab(NOTE, note, r"knows the cell-type layer exactly and nothing else: cell type ([0-9.]+) \[([0-9.]+), ([0-9.]+)\], "
           r"experiment ([0-9.]+), donor ([0-9.]+)", 5)
ef2 = grab(NOTE, note, r"also knows the experiment layer: cell type ([0-9.]+) \[([0-9.]+), ([0-9.]+)\], experiment "
           r"([0-9.]+) \[([0-9.]+), ([0-9.]+)\], donor ([0-9.]+)", 7)
ef = dict(ct=dict(cell=[num(x) for x in ef1[:3]], exp=num(ef1[3]), donor=num(ef1[4])),
          ctx=dict(cell=[num(x) for x in ef2[:3]], exp=[num(x) for x in ef2[3:6]], donor=num(ef2[6])))

# --- real data today: counts only, no layer values ---------------------------------------
ds = text("results/t183/DATASET_EIC_V1.md")
n_rung_lines = int(grab("results/t183/DATASET_EIC_V1.md", ds, r"The ([0-9,]+) rung lines are in `rung_pairs.tsv`"))
tr = grab("results/t183/DATASET_EIC_V1.md", ds, r"\| training \| ([0-9]+) \| ([0-9]+) \|", 2)
va = grab("results/t183/DATASET_EIC_V1.md", ds, r"\| validation \| ([0-9]+) \| ([0-9]+) \|", 2)
rows = list(csv.DictReader(text("results/t183/rung_pairs.tsv").splitlines(), delimiter="\t"))
if len(rows) != n_rung_lines:
    sys.exit("build_deck: rung_pairs.tsv line count differs from DATASET_EIC_V1.md")
RUNGS = ["pseudoreplicate", "sequencing", "technical", "isogenic", "anisogenic", "other_experiment", "other_cell_type"]
pairs_per_rung = {r: sum(int(x["n_pairs"]) for x in rows if x["rung"] == r) for r in RUNGS}
n_assays = len({x["assay"] for x in rows})
sub = text("results/t185/SUBMITTED.md")
ev = grab("results/t185/SUBMITTED.md", sub, r"([0-9,]+) correction pairs \(0 dropped\); ([0-9,]+) replicate pairs", 2)
n_regions = int(grab("results/t185/SUBMITTED.md", sub, r"Content: ([0-9,]+) regions of"))
NOTEPAD = "INSTRUCTIONS/peeling_the_onion/long-horizon notepad.md"
npd = text(NOTEPAD)
sec = grab(NOTEPAD, npd, r"second experiment for ([0-9]+) of the ([0-9]+) non-control EIC tracks", 2)

data = dict(
    sim=sim, defects=defects, audit=dict(rows_before=n_before_rows, defects_before=n_before_def,
                                         rows_after=n_after_rows, defects_after=n_after_def),
    h7=h7, crossed=crossed, nested_exp=dict(planted=nested_exp["planted"], est=nested_exp["est"],
                                            lo=nested_exp["lo"], hi=nested_exp["hi"]),
    ident=ident, budget_max=budget_max, ef=ef,
    real=dict(rung_lines=n_rung_lines, n_assays=n_assays, pairs=pairs_per_rung,
              train=[int(x) for x in tr], val=[int(x) for x in va],
              corr_pairs=num(ev[0]), rep_pairs=num(ev[1]), regions=n_regions,
              second_exp=[int(x) for x in sec]))

read += [HERE / "build_deck.py", HERE / "deck_src.html", HERE / "simulate_figures.py"]
sources = {p.relative_to(VAULT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in read}
src = (HERE / "deck_src.html").read_bytes().decode("utf-8")
out = src.replace("__SOURCES__", json.dumps(sources, sort_keys=True))
out = out.replace("__DATA__", json.dumps(data, separators=(",", ":")))
(HERE / "index.html").write_bytes(out.encode("utf-8"))
print(f"build_deck: wrote {HERE / 'index.html'} from {len(sources)} files")
