"""Compare verify_raw.json (stdlib, from the raw run files) with analyze.py's tables -> verify.txt.

One AGREE or DISAGREE line per headline (tolerance 1e-9 on values and spreads, exact on counts,
flags and readings), then one DISAGREE line per failing cell. A cell present in only one method
is DISAGREE. Headlines: m5 depth-law medians and spreads (class all; scopes all7, darm2_tracks),
m1 held-out pooled CRPS with its oracle_scaled / scale_error split (knob all, class all), m3
trained base<->arm pooled CRPS, the n counts of the grid arms, and the readings of c1-c5 and the
combination rule.

    python verify_compare.py [<dir>]
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

TOL = 1e-9


def main(here):
    H = Path(here)
    v = json.load(open(H / "verify_raw.json"))
    lines, bad = [], []

    def same(x, y, exact):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return y is None or (isinstance(y, float) and math.isnan(y))
        if y is None or (isinstance(y, float) and math.isnan(y)):
            return False
        return x == y if exact else abs(float(x) - float(y)) <= TOL

    def check(headline, raw, got, exact=False):
        n, worst = 0, 0.0
        for k in sorted(set(raw) | set(got)):
            if k not in raw or k not in got:
                bad.append(f"DISAGREE {headline} {k}: in {'analyze' if k in got else 'verify_raw'} only")
                continue
            n += 1
            x, y = raw[k], got[k]
            if not exact and x is not None and y is not None and not (isinstance(y, float) and math.isnan(y)):
                worst = max(worst, abs(float(x) - float(y)))
            if not same(x, y, exact):
                bad.append(f"DISAGREE {headline} {k}: verify {x!r} analyze {y!r}")
        ok = n > 0 and not any(b.startswith(f"DISAGREE {headline} ") for b in bad)
        lines.append(f"{'AGREE' if ok else 'DISAGREE'} {headline}: {n} cells, max |difference| {worst:.2e}")

    def nan(x):
        return None if pd.isna(x) else float(x)

    m5 = pd.read_csv(H / "m5_depthlaw_table.tsv", sep="\t")
    m5 = m5[(m5["class"] == "all") & m5.group.isin(["trained_base", "heldout_base", "heldout_r2", "trained_r4"])]
    k5 = [f"{r.scope}|{r.arm}|{r.group}" for _, r in m5.iterrows()]
    check("m5 depth-law median |log2 error|", {k: x["value"] for k, x in v["m5"].items()},
          dict(zip(k5, m5.median_abs_err)))
    check("m5 seed spread", {k: x["spread"] for k, x in v["m5"].items()},
          dict(zip(k5, map(nan, m5.seed_spread))))
    check("m5 complete flag", {k: x["complete"] for k, x in v["m5"].items()},
          dict(zip(k5, map(bool, m5.complete))), exact=True)
    m1 = pd.read_csv(H / "m1_heldout_table.tsv", sep="\t")
    m1 = m1[(m1["class"] == "all") & (m1.subgroup == "heldout75") & (m1.knob == "all")]
    k1 = [f"{r.scope}|{r.arm}" for _, r in m1.iterrows()]
    for name, col, key in (("crps_all", "crps_all", "value"),
                           ("oracle_scaled", "crps_oracle_scaled_all", "oracle_scaled"),
                           ("scale_error", "scale_error_all", "scale_error"),
                           ("seed spread", "seed_spread", "spread")):
        check(f"m1 held-out pooled {name}", {k: x[key] for k, x in v["m1"].items()},
              dict(zip(k1, map(nan, m1[col]))))
    check("m1 complete flag", {k: x["complete"] for k, x in v["m1"].items()},
          dict(zip(k1, map(bool, m1.complete))), exact=True)
    m3 = pd.read_csv(H / "m3_trained_table.tsv", sep="\t")
    m3 = m3[(m3["class"] == "all") & (m3.group == "trained_base")]
    k3 = [f"{r.scope}|{r.arm}" for _, r in m3.iterrows()]
    check("m3 trained pooled crps_all", {k: x["value"] for k, x in v["m3"].items()},
          dict(zip(k3, m3.crps_all)))
    check("m3 seed spread", {k: x["spread"] for k, x in v["m3"].items()},
          dict(zip(k3, map(nan, m3.seed_spread))))
    nt = pd.read_csv(H / "n_table.tsv", sep="\t").set_index("arm")
    cols = {"run_dirs": "run_dirs", "finished": "finished", "trained": "TRAIN_DONE",
            "train75": "n_runs_train75", "extra75": "n_runs_extra75", "extra_ok": "n_extra_ok",
            "configs": "n_configs", "best_last10": "n_best_last10"}
    raw, got = {}, {}
    for arm in ("Darm1", "Darm2"):
        for a, b in cols.items():
            raw[f"{arm}|{a}"] = int(v["n"][arm][a])
            got[f"{arm}|{a}"] = int(nt.loc[arm, b])
    raw["Dref|best_last10"] = int(v["n"]["Dref"]["best_last10"])
    got["Dref|best_last10"] = int(nt.loc["Dref", "n_best_last10"])
    check("n counts (runs, leaks, extra pairs, best step)", raw, got, exact=True)
    ck = pd.read_csv(H / "checks_readings.tsv", sep="\t")

    def rd(c, arm, items):
        s = ck[(ck.check == c) & (ck.arm == arm) & ck["item"].isin(items)]
        return s.reading.iloc[0]

    got = {"c1": rd("c1", "Darm1", ["gap closed on heldout_r2"]),
           "c2|Darm1": rd("c2", "Darm1", ["arm reading"]), "c2|Darm2": rd("c2", "Darm2", ["arm reading"]),
           "c2": rd("c2", "both", ["check reading"]),
           "c3": rd("c3", "Darm1", ["heldout_r2 vs trained_r4 + 2 x spread"]),
           "c4": rd("c4", "Darm1", ["check reading"]),
           "c5|Darm1": rd("c5", "Darm1", ["arm reading"]), "c5|Darm2": rd("c5", "Darm2", ["arm reading"]),
           "c5": rd("c5", "both", ["check reading"]), "rule": rd("rule", "Darm1", ["rule reading"])}
    raw = {k: v["readings"][k] for k in got}
    check("readings c1-c5 and the rule", raw, got, exact=True)
    out = lines + bad
    (H / "verify.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent)
