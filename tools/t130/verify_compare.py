"""t130: compare verify_raw.json (stdlib second method) with analyze.py's blind_table.tsv.

One AGREE / DISAGREE line per headline (|difference| <= 1e-6), then one DISAGREE line per failing
cell. A value missing on either side is DISAGREE. Stdlib only. Writes <results_dir>/verify.txt.

    python3 verify_compare.py <results_dir>
"""
import csv
import json
import os
import sys

TOL = 1e-6
KEY = ["space", "knob", "mark_class", "rung", "g_version", "direction", "subset"]


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def main(argv):
    if len(argv) != 1:
        print("usage: verify_compare.py <results_dir>", file=sys.stderr)
        return 2
    d = argv[0]
    with open(os.path.join(d, "verify_raw.json")) as fh:
        v = json.load(fh)
    with open(os.path.join(d, "blind_table.tsv"), newline="") as fh:
        t = {"|".join(r[k] for k in KEY): r for r in csv.DictReader(fh, delimiter="\t")}
    lines, bad = [], []
    for col, cells in v.items():
        worst, n, ok = 0.0, 0, True
        for k, x in cells.items():
            y = num(t[k][col]) if k in t else None
            if x is None or y is None:
                ok = False
                bad.append(f"DISAGREE {col} {k}: verify {x!r} analyze {y!r}")
                continue
            n += 1
            dd = abs(x - y)
            worst = max(worst, dd)
            if dd > TOL:
                ok = False
                bad.append(f"DISAGREE {col} {k}: verify {x!r} analyze {y!r}")
        ok = ok and n > 0
        lines.append(f"{'AGREE' if ok else 'DISAGREE'} {col}: {n} cells, max |difference| {worst:.2e}")
    text = "\n".join(lines + bad) + "\n"
    with open(os.path.join(d, "verify.txt"), "w") as fh:
        fh.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
