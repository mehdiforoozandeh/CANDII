#!/bin/bash
# t396 C12b: dedup at 1 bp, collect and decide the dedup unit rule. No arguments.
# Every step is skipped when its output exists. Pulls the small sc_1bp outputs from Fir.
set -euo pipefail

FIR_OUT=/project/def-maxwl/mforooz/t396/dedup_1bp      # sc_1bp outputs: <experiment>/{unit,shape}_<label>.json
JOB_IDS="6727806 6729250 6733480 6733481 6733482 6733483 6733484"   # sc_1bp jobs of this chunk (probe OOM, probe 2, then the other 5)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
N_FULL_BUILD=2811                                      # measurement folders of the read-start store /scratch/mforooz/rs392/store
cd "$HERE"
PY="$(command -v python3 || command -v python)"

# 1. pull the per-unit jsons (small) from Fir
if [ ! -f fir/.pulled ]; then
  mkdir -p fir
  rsync -a --prune-empty-dirs -e 'ssh -o BatchMode=yes' --include='*/' --include='unit_*.json' --include='shape_*.json' \
    --exclude='*' "fir:$FIR_OUT/" fir/
  touch fir/.pulled
fi

# 2. measured cost of the sc_1bp jobs (sacct), and the cost of a full 1 bp build
#    full build = mean (and median) CPU-h per experiment over the COMPLETED jobs x N_FULL_BUILD
if [ ! -f cost.tsv ]; then
  ids="$(echo $JOB_IDS | tr ' ' ',')"
  ssh -o BatchMode=yes fir "sacct -n -P -j $ids -o JobID,State,ElapsedRaw,TotalCPU,MaxRSS,AllocCPUS" 2>/dev/null \
    | awk -F'|' '$1 ~ /\.batch$/' > cost.sacct.tmp
  "$PY" - "$N_FULL_BUILD" <<'EOF2'
import statistics, sys
n_full = int(sys.argv[1])
def secs(t):
    d = 0
    if "-" in t: d, t = t.split("-"); d = int(d)
    p = [float(x) for x in t.split(":")]
    while len(p) < 3: p.insert(0, 0.0)
    return d * 86400 + p[0] * 3600 + p[1] * 60 + p[2]
exp = {}
for line in open("jobs.tsv"):
    f = line.rstrip("\n").split("\t")
    if len(f) > 4 and "EXP=" in f[4]:
        exp[f[0]] = f[4].split("EXP=")[1].split()[0]
cols = ["row_type", "job_id", "experiment", "state", "elapsed_s", "total_cpu_s", "cpu_h", "max_rss_kb", "alloc_cpus"]
rows, done = [], []
for line in open("cost.sacct.tmp"):
    j, st, el, cpu, rss, nc = line.rstrip("\n").split("|")
    j = j.split(".")[0]; c = secs(cpu)
    rows.append(["job", j, exp.get(j, ""), st, el, "%.0f" % c, "%.4f" % (c / 3600), rss.rstrip("K"), nc])
    if st == "COMPLETED": done.append(c / 3600)
mean, med = statistics.mean(done), statistics.median(done)
rows.append(["mean_cpu_h_per_experiment", "", "%d completed" % len(done), "", "", "", "%.4f" % mean, "", ""])
rows.append(["median_cpu_h_per_experiment", "", "%d completed" % len(done), "", "", "", "%.4f" % med, "", ""])
rows.append(["full_build_cpu_h_mean_x%d" % n_full, "", "", "", "", "", "%.1f" % (mean * n_full), "", ""])
rows.append(["full_build_cpu_h_median_x%d" % n_full, "", "", "", "", "", "%.1f" % (med * n_full), "", ""])
with open("cost.tsv", "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in rows: f.write("\t".join(r) + "\n")
EOF2
  rm -f cost.sacct.tmp
fi

# 3. the unit rule (dedup_1bp.py rule, C12a)
if [ ! -f UNIT_RULE.json ]; then
  "$PY" "$REPO/tools/t396/dedup_1bp.py" rule --units fir/*/unit_*.json --out UNIT_RULE.json > /dev/null
fi

# 4. mechanism.tsv (one row per unit json) and shape.tsv (one row per shape json)
if [ ! -f mechanism.tsv ] || [ ! -f shape.tsv ]; then
  "$PY" - <<'EOF'
import glob, json
cols = ["experiment", "label", "n_units", "n_off_lines", "n_off_distinct", "n_on_lines",
        "n_on_distinct", "n_on_absent_from_off", "n_off_absent_from_on", "share_on_absent_from_off",
        "share_off_absent_from_on", "share_on_repeated_lines", "share_off_duplicate_lines"]
rows = [json.load(open(f)) for f in sorted(glob.glob("fir/*/unit_*.json"))]
with open("mechanism.tsv", "w") as f:
    f.write("\t".join(cols + ["on_equals_distinct_off"]) + "\n")
    for u in rows:
        eq = max(u["share_on_absent_from_off"], u["share_off_absent_from_on"],
                 u["share_on_repeated_lines"]) <= 1e-3
        f.write("\t".join(str(u[c]) for c in cols) + "\t" + str(eq).lower() + "\n")
scol = ["experiment", "label", "n_units", "dirichlet_a_1bp", "dirichlet_a_25bp_caps_fit_shape",
        "K_25bp", "n_top_strand_bins", "top1pct_sum_off", "top1pct_sum_distinct_off", "top1pct_sum_on"]
laws = ["dm_distinct_a_25bp", "dm_distinct_a_1bp", "occupancy"]
for l in laws:
    scol += [f"{l}_rel_err_vs_distinct_off", f"{l}_rel_err_vs_on", f"{l}_deviance_distinct_off"]
scol += ["share_off_positions_with_1_read", "share_off_positions_with_2plus_reads", "max_off_reads_per_position"]
with open("shape.tsv", "w") as f:
    f.write("\t".join(scol) + "\n")
    for p in sorted(glob.glob("fir/*/shape_*.json")):
        s = json.load(open(p)); t = s["top1pct"]
        h = {int(k): v for k, v in s["hist_off_reads_per_position"].items()}
        tot = sum(h.values())
        r = [s["experiment"], s["label"], s["n_units"], "%.4g" % s["dirichlet_a_1bp"],
             "%.4g" % s["dirichlet_a_25bp_caps_fit_shape"], s["K_25bp"], s["n_top_strand_bins"],
             t["sum_off"], t["sum_distinct_off"], t["sum_on"]]
        for l in laws:
            r += ["%.4g" % t[l]["rel_err_vs_distinct_off"], "%.4g" % t[l]["rel_err_vs_on"],
                  "%.6g" % t[l]["deviance_distinct_off"]]
        r += ["%.4f" % (h.get(1, 0) / tot), "%.4f" % (1 - h.get(1, 0) / tot), max(h)]
        f.write("\t".join(str(x) for x in r) + "\n")
EOF
fi
echo "dedup_1bp: $(cat UNIT_RULE.json | tr -d '\n ')"
