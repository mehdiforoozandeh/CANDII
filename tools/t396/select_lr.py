"""t396 iteration 2, gate (f): pick the learning rate of every factorial arm.

    python tools/t396/select_lr.py --runs-root <dir> --out sweep.tsv

Reads `<runs-root>/<arm>__lr<lr>/seed<k>/log.jsonl`. A run counts when `DONE` exists, `CAPPED`
does not, and its log holds at least 10 `heldout_loss` entries; its score is the mean of the last
10. Other runs are left out with a note on stderr.

One row per (arm, lr), columns:

  arm, lr
  heldout_per_seed   `seed<k>:<score>` joined by `;`
  heldout_mean       mean over the counted seeds
  seed_range         max - min over the counted seeds
  n_seeds            counted seeds
  edge_pick          on the picked row: `yes` when its lr is the lowest or highest lr tried for
                     the arm (every `<arm>__lr*` folder), else `no`; empty on other rows
  status             `picked` on the row with the lowest heldout_mean of its arm, else empty

An arm whose lr rows do not all hold the same, non-zero number of counted seeds gets no pick
(stderr says why) and the exit code is 1. Extended rates (gate f: 1e-4 below, 1e-2 above) are
just more `<arm>__lr<lr>` folders: rerun after they finish and the pick covers them.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

LAST_N = 10
DIR_RE = re.compile(r"^(?P<arm>.+)__lr(?P<lr>[0-9.eE+-]+)$")
SEED_RE = re.compile(r"^seed(?P<k>\d+)$")
COLUMNS = ("arm", "lr", "heldout_per_seed", "heldout_mean", "seed_range", "n_seeds",
           "edge_pick", "status")


def run_score(run: Path) -> float | None:
    """Mean of the last LAST_N `heldout_loss` entries of a finished run, else None."""
    if not (run / "DONE").is_file() or (run / "CAPPED").exists():
        return None
    log = run / "log.jsonl"
    if not log.is_file():
        return None
    vals = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if "heldout_loss" in rec:
                vals.append(float(rec["heldout_loss"]))
    if len(vals) < LAST_N:
        return None
    return sum(vals[-LAST_N:]) / LAST_N


def sweep(runs_root) -> tuple[list[dict], list[str]]:
    """(rows, problems) for every `<arm>__lr<lr>` folder under runs_root."""
    root = Path(runs_root)
    groups: dict[str, list] = {}
    problems = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = DIR_RE.match(d.name)
        if not m:
            continue
        scores = {}
        for s in sorted(d.iterdir()):
            k = SEED_RE.match(s.name)
            if not (k and s.is_dir()):
                continue
            v = run_score(s)
            if v is None:
                problems.append(f"{d.name}/{s.name}: not counted (no DONE, CAPPED, or fewer "
                                f"than {LAST_N} heldout_loss entries)")
            else:
                scores[int(k["k"])] = v
        groups.setdefault(m["arm"], []).append((m["lr"], scores))
    rows = []
    for arm, items in sorted(groups.items()):
        items.sort(key=lambda t: float(t[0]))
        arm_rows = []
        for lr, sc in items:
            vals = [sc[k] for k in sorted(sc)]
            arm_rows.append({
                "arm": arm, "lr": lr,
                "heldout_per_seed": ";".join(f"seed{k}:{sc[k]:.6g}" for k in sorted(sc)),
                "heldout_mean": f"{sum(vals) / len(vals):.6g}" if vals else "",
                "seed_range": f"{max(vals) - min(vals):.6g}" if vals else "",
                "n_seeds": len(vals), "edge_pick": "", "status": "",
                "_mean": sum(vals) / len(vals) if vals else None})
        ns = {r["n_seeds"] for r in arm_rows}
        if len(ns) != 1 or 0 in ns:
            problems.append(f"{arm}: no pick, counted seeds per lr = "
                            f"{[(r['lr'], r['n_seeds']) for r in arm_rows]}")
        else:
            best = min(arm_rows, key=lambda r: r["_mean"])
            best["status"] = "picked"
            best["edge_pick"] = "yes" if best in (arm_rows[0], arm_rows[-1]) else "no"
        for r in arm_rows:
            del r["_mean"]
        rows.extend(arm_rows)
    return rows, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows, problems = sweep(a.runs_root)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for p in problems:
        print(f"select_lr: {p}", file=sys.stderr)
    for r in rows:
        if r["status"] == "picked":
            print(f"{r['arm']}\tlr={r['lr']}\tmean={r['heldout_mean']}\tedge_pick={r['edge_pick']}")
    return 1 if any(": no pick," in p for p in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
