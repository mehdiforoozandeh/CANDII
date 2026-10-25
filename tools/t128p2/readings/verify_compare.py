"""Compare verify_raw.json (stdlib, from the raw run files) with analyze.py's tables -> verify.txt.

One AGREE or DISAGREE line per headline (tolerance 1e-6 on means and medians, exact on counts),
then one DISAGREE line per failing cell, if any. A cell present in only one method is DISAGREE.
Headlines: pooled never-trained CRPS (+ its oracle_scaled / scale_error split) per arm x gv,
pooled trained CRPS per arm x gv, depth-law medians per arm x gv x group, DSanchor's hop-1
pooled CRPS per gv x group, the metric-2 gen_gap (knob all) per arm x gv, the noise label of
DSfree vs D on metric 1 (never, knob all) per gv, and the metric-9 counts and share per arm x gv.
"""
import json
from pathlib import Path

import pandas as pd

H = Path(__file__).resolve().parent
TOL = 1e-6
v = json.load(open(H / "verify_raw.json"))


def tab(name, keys, **sel):
    t = pd.read_csv(H / name, sep="\t")
    for c, x in sel.items():
        t = t[t[c] == x]
    return t.set_index(keys)


lines, bad = [], []


def check(headline, raw, t, col, exact=False):
    worst, n, miss, seen = 0.0, 0, [], set()
    for k, x in raw.items():
        key = tuple(k.split("|"))
        seen.add(key)
        if key not in t.index:
            miss.append(k)
            continue
        y = t.loc[key, col]
        y = float(y.iloc[0]) if isinstance(y, pd.Series) else float(y)
        n += 1
        dd = abs(x - y)
        worst = max(worst, dd)
        if not (dd == 0 if exact else dd <= TOL):
            bad.append(f"DISAGREE {headline} {k}: verify {x!r} analyze {y!r}")
    extra = [i for i in t.index if i not in seen]
    for k in miss:
        bad.append(f"DISAGREE {headline} {k}: in verify_raw only")
    for k in extra:
        bad.append(f"DISAGREE {headline} {'|'.join(map(str, k))}: in analyze only")
    ok = (worst == 0 if exact else worst <= TOL) and not miss and not extra and n > 0
    lines.append(f"{'AGREE' if ok else 'DISAGREE'} {headline}: {n} cells, max |difference| {worst:.2e}")


m1 = tab("m1_never_table.tsv", ["arm", "g_version"], subgroup="never", knob="all")
check("metric 1 pooled never-trained CRPS (knob all)", v["never_crps_pooled"], m1, "crps_all")
check("metric 1 split: oracle_scaled", v["never_oracle_scaled_pooled"], m1, "crps_oracle_scaled_all")
check("metric 1 split: scale_error", v["never_scale_error_pooled"], m1, "scale_error_all")
check("metric 3 pooled trained CRPS (knob all)", v["trained_crps_pooled"],
      tab("m3_trained_table.tsv", ["arm", "g_version"], knob="all"), "crps_all")
check("metric 5 depth-law median |log2 error|", v["depth_median_abs_err"],
      tab("m5_depthlaw_table.tsv", ["arm", "g_version", "group"]), "median_abs_err")
check("metric 6 DSanchor hop-1 pooled CRPS (encoder, knob all)", v["hop1_crps_pooled"],
      tab("m6_hop1_table.tsv", ["arm", "g_version", "group"], knob="all",
          source="encoder NB head (hop1.json)"), "crps_all")
check("metric 2 gen_gap (knob all: same-knob never minus trained)", v["gen_gap_all"],
      tab("m2_gap_table.tsv", ["arm", "g_version"], knob="all"), "gen_gap")
lab = tab("m1_never_table.tsv", ["arm", "g_version"], subgroup="never", knob="all")
want = v["label_DSfree_vs_D_never"]
got = {f"{a}|{g}": r.label_vs_D for (a, g), r in lab.iterrows() if a == "DSfree"}
ok = bool(want) and want == got
lines.append(f"{'AGREE' if ok else 'DISAGREE'} noise label DSfree vs D, metric 1 never (knob all): "
             f"verify {want} | analyze {got}")
m9 = tab("m9_steps_table.tsv", ["arm", "g_version"])
for col, exact in (("n_runs", True), ("n_best_last10", True), ("n_at_cap", True),
                   ("share_best_last10", False)):
    check(f"metric 9 {col}", {k: r[col] for k, r in v["m9"].items()}, m9, col, exact=exact)
out = lines + bad
(H / "verify.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
