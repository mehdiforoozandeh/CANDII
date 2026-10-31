"""Compare verify_raw.json (stdlib, from the raw run files) with analyze.py's tables -> verify.txt.

One AGREE or DISAGREE line per headline (tolerance 1e-6 on every value, exact on the lambda
choice), then one DISAGREE line per failing cell, if any. Cells present in only one method are
DISAGREE too (old D from t126 is not recomputed and is not compared).
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
    t = t[t.arm != "D"]
    return t.set_index(keys)


lines, bad = [], []


def check(headline, raw, t, col, split=None):
    worst, n, miss = 0.0, 0, []
    seen = set()
    for k, x in raw.items():
        key = tuple(k.split("|")) if split is None else split(k)
        seen.add(key)
        if key not in t.index:
            miss.append(k)
            continue
        y = t.loc[key, col]
        y = float(y.iloc[0]) if isinstance(y, pd.Series) else float(y)
        n += 1
        d = abs(x - y)
        worst = max(worst, d)
        if not d <= TOL:
            bad.append(f"DISAGREE {headline} {k}: verify {x!r} analyze {y!r}")
    extra = [i for i in t.index if i not in seen]
    for k in miss:
        bad.append(f"DISAGREE {headline} {k}: in verify_raw only")
    for k in extra:
        bad.append(f"DISAGREE {headline} {'|'.join(map(str, k))}: in analyze only")
    ok = worst <= TOL and not miss and not extra
    lines.append(f"{'AGREE' if ok else 'DISAGREE'} {headline}: {n} cells, max |difference| {worst:.2e}")


check("depth median |log2 error| (reading 1)", v["depth_median_abs_err"],
      tab("depth_table.tsv", ["arm", "g_version", "group"]), "median_abs_err")
check("pooled trained CRPS, real", v["trained_crps_pooled"],
      tab("fit_table.tsv", ["arm", "g_version"], model="real"), "crps_all")
check("pooled never-trained CRPS, real (reading 2)", v["never_crps_pooled"],
      tab("knob_table.tsv", ["arm", "g_version"], group="never", knob="all"), "crps_all")
check("median gain over the twin, never-trained (reading 3)", v["twin_gain_never_median"],
      tab("twin_table.tsv", ["arm", "g_version"], group="never", knob="all"), "gain_median")
check("collapse median |Z - x|, real (reading 4)", v["collapse_median_abs_z_minus_x"],
      tab("collapse_table.tsv", ["arm", "g_version"], model="real", knob="all"),
      "median_abs_z_minus_x")
check("collapse median corr(Z, x), real (reading 4)", v["collapse_median_corr"],
      tab("collapse_table.tsv", ["arm", "g_version"], model="real", knob="all"), "corr_z_x")
check("mean best val NLL, TRAIN_DONE vs timing.json (lambda rule)", v["val_nll_mean"],
      tab("lambda_table.tsv", ["arm", "g_version"]), "val_nll_mean")
ch = json.load(open(H / "lambda_chosen.json"))
lam_a = {gv: r["arm"] for gv, r in ch.items()}
lam_v = {gv: a for gv, a in v["lambda"].items() if a is not None}
lines.append(f"{'AGREE' if lam_a == lam_v else 'DISAGREE'} lambda choice: verify {lam_v} | analyze {lam_a}")
out = lines + bad
(H / "verify.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
