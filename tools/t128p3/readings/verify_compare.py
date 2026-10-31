"""Compare verify_raw.json (stdlib, from the raw run files) with analyze.py's tables -> verify.txt.

One AGREE or DISAGREE line per headline (tolerance 1e-6 on means and medians, exact on counts),
then one DISAGREE line per failing cell, if any. A cell present in only one method is DISAGREE.
Only the phase-3 arms are compared (Dbigg, Dbigf, Dbiggviabase, Dbigfviabase); the reference rows
of analyze.py's tables are phase 2's, verified there.
Headlines: pooled never-trained CRPS (+ its oracle_scaled / scale_error split) per arm x gv, the
via-base never pair count per arm x gv, pooled trained CRPS per direct arm x gv, depth-law
medians per arm x gv x group, and the metric-9 counts, share and best-step median per arm x gv.

    python verify_compare.py [<dir>]      (default: beside this file)
"""
import json
import sys
from pathlib import Path

import pandas as pd

TOL = 1e-6
PHASE3 = ("Dbigg", "Dbigf", "Dbiggviabase", "Dbigfviabase")


def main(h):
    h = Path(h)
    v = json.load(open(h / "verify_raw.json"))
    lines, bad = [], []

    def tab(name, keys, **sel):
        t = pd.read_csv(h / name, sep="\t")
        t = t[t.arm.isin(PHASE3)]
        for c, x in sel.items():
            t = t[t[c] == x]
        return t.set_index(keys)

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
        lines.append(f"{'AGREE' if ok else 'DISAGREE'} {headline}: {n} cells, "
                     f"max |difference| {worst:.2e}")

    m1 = tab("m1_never_table.tsv", ["arm", "g_version"], subgroup="never", knob="all")
    check("metric 1 pooled never-trained CRPS (knob all)", v["never_crps_pooled"], m1, "crps_all")
    check("metric 1 split: oracle_scaled", v["never_oracle_scaled_pooled"], m1,
          "crps_oracle_scaled_all")
    check("metric 1 split: scale_error", v["never_scale_error_pooled"], m1, "scale_error_all")
    check("via-base never-trained pair count (knob all)", v["viabase_never_n_pairs"],
          m1[m1.index.get_level_values("arm").str.endswith("viabase")], "n_pairs", exact=True)
    check("metric 3 pooled trained CRPS (knob all)", v["trained_crps_pooled"],
          tab("m3_trained_table.tsv", ["arm", "g_version"], knob="all"), "crps_all")
    check("metric 5 depth-law median |log2 error|", v["depth_median_abs_err"],
          tab("m5_depthlaw_table.tsv", ["arm", "g_version", "group"]), "median_abs_err")
    m9 = tab("m9_steps_table.tsv", ["arm", "g_version"])
    for col, exact in (("n_runs", True), ("n_best_last10", True), ("n_at_cap", True),
                       ("n_best_gt12000", True), ("share_best_last10", False),
                       ("best_step_median", False)):
        check(f"metric 9 {col}", {k: r[col] for k, r in v["m9"].items()}, m9, col, exact=exact)
    out = lines + bad
    (h / "verify.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent)
