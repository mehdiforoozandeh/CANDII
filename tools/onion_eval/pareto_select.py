"""Pareto selection of miniCANDI recipe arms against the three base models (decision D1, D21).

Input: one or more TSVs written by `select_rule.py arm-summary` (columns `arm`, `n_seeds`,
`<term>.mean`, `<term>.range`). The three terms are imputation_raw, correction_v3_raw and
uncertainty_raw. For term k, best_k is the reference (base) row with the highest mean, and
floor_k = best_k.mean - best_k.range (range = max - min over seeds).

An arm is eligible when its mean is >= floor_k on all three terms. Term k is beyond when
arm.mean - best_k.mean > max(arm.range, best_k.range). An arm wins when it is eligible and at
least one term is beyond. z_k = (arm.mean - best_k.mean) / best_k.range; score = min_k z_k.
The winner of a model is its winning arm with the highest score; the overall winner is the same
over all models. The model of an arm is its name suffix (_convae, _convattn, _unet).

Class: PASS = wins; NEAR = eligible with no beyond term, or exactly one term below its floor by
less than the arm's own range; FAIL otherwise; WAIT = fewer than --min-seeds seeds (never wins).

`--depth-native-summary`: for an arm with a row in that table, the eligibility test (and the
NEAR margin) uses that row; beyond, z and score use the main row.

Exit 0: an overall winner exists. 1: none. 3: a reference row is missing, has fewer than
--min-seeds seeds, or has an undefined term. 2: bad input (duplicate arm across summaries).

    python tools/onion_eval/pareto_select.py --summary arm_summary.tsv [--summary ...] \\
        [--ref convae=eic_v1_2_rung2_base_heldout ...] [--arm-prefix rcp_] [--min-seeds 3] \\
        [--depth-native-summary native.tsv] [--out selection.tsv]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

TERMS = ("imputation_raw", "correction_v3_raw", "uncertainty_raw")
MODELS = ("convae", "convattn", "unet")
DEFAULT_REFS = {"convae": "eic_v1_2_rung2_base_heldout",
                "convattn": "eic_v1_2_rung3_base_heldout",
                "unet": "eic_v1_2_rung4_base_heldout"}
EPS = 1e-9          # the TSVs carry 4 decimals; absorb float error in floor = mean - range


class RefError(Exception):
    """A reference row is missing, short of seeds, or has an undefined term (exit 3)."""


def _num(x):
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
    else:
        s = str(x).strip()
        if not s:
            return None
        try:
            v = float(s)
        except ValueError:
            return None
    return v if math.isfinite(v) else None


def _seeds(r: dict) -> int:
    v = _num(r.get("n_seeds"))
    return 0 if v is None else int(v)


def _mr(r: dict, k: str):
    return _num(r.get(f"{k}.mean")), _num(r.get(f"{k}.range"))


def model_of(arm: str):
    for m in MODELS:
        if arm.endswith(f"_{m}"):
            return m
    return None


def floors(rows: dict, refs: dict, min_seeds: int) -> dict:
    """Per term: best reference (model, mean, range) and floor. Raises RefError."""
    for m, name in refs.items():
        r = rows.get(name)
        if r is None:
            raise RefError(f"reference {m}={name}: no row in the summaries")
        if _seeds(r) < min_seeds:
            raise RefError(f"reference {m}={name}: {_seeds(r)} seeds < {min_seeds}")
        for k in TERMS:
            mean, rng = _mr(r, k)
            if mean is None or rng is None:
                raise RefError(f"reference {m}={name}: {k} undefined")
    out = {}
    for k in TERMS:
        m = max(refs, key=lambda m: _mr(rows[refs[m]], k)[0])
        mean, rng = _mr(rows[refs[m]], k)
        out[k] = {"best_model": m, "best_arm": refs[m], "best_mean": mean, "best_range": rng,
                  "floor": mean - rng}
    return out


def _z(diff: float, rng: float) -> float:
    if rng > 0:
        return diff / rng
    return 0.0 if diff == 0 else math.copysign(math.inf, diff)


def evaluate(r: dict, fl: dict, rows: dict, refs: dict, min_seeds: int, native: dict | None) -> dict:
    arm = r["arm"]
    model = model_of(arm)
    o = {"arm": arm, "model": model or "", "n_seeds": _seeds(r)}
    elig_row = native if native is not None else r
    o["eligibility_row"] = "depth_native" if native is not None else "main"
    defined = True
    below = []                      # terms below the floor in the eligibility row
    beyond = []
    zs = []
    for k in TERMS:
        mean, rng = _mr(r, k)
        o[f"{k}.mean"], o[f"{k}.range"] = mean, rng
        emean, erng = _mr(elig_row, k)
        f = fl[k]
        if mean is None or rng is None or emean is None or erng is None:
            defined = False
            o[f"{k}.z"] = None
        else:
            if emean < f["floor"] - EPS:
                below.append((k, f["floor"] - emean, erng))
            diff = mean - f["best_mean"]
            if diff > max(rng, f["best_range"]) + EPS:
                beyond.append(k)
            z = _z(diff, f["best_range"])
            o[f"{k}.z"] = z
            zs.append(z)
        # information only: difference to the arm's own base and whether it is beyond
        own = rows.get(refs.get(model, "")) if model else None
        om, orng = _mr(own, k) if own else (None, None)
        if mean is None or rng is None or om is None or orng is None:
            o[f"own_diff.{k}"], o[f"own_beyond.{k}"] = None, None
        else:
            o[f"own_diff.{k}"] = mean - om
            o[f"own_beyond.{k}"] = (mean - om) > max(rng, orng) + EPS
    eligible = defined and not below
    o["eligible"] = eligible
    o["beyond_terms"] = ",".join(beyond) if defined else ""
    o["score"] = min(zs) if defined else None
    wait = o["n_seeds"] < min_seeds
    o["wins"] = bool(eligible and beyond and not wait)
    if wait:
        cls = "WAIT"
    elif o["wins"]:
        cls = "PASS"
    elif eligible:
        cls = "NEAR"
    elif defined and len(below) == 1 and below[0][1] < below[0][2]:
        cls = "NEAR"
    else:
        cls = "FAIL"
    o["class"] = cls
    return o


def _best(arms: list[dict]):
    w = [a for a in arms if a["wins"]]
    return max(w, key=lambda a: a["score"])["arm"] if w else None


def select(rows: list[dict], refs: dict[str, str], min_seeds: int = 3,
           depth_native: list[dict] | None = None) -> dict:
    """Apply D1. `rows`: arm-summary rows (reference rows included). Every non-reference row
    is an arm. Raises RefError when the references cannot define the floors."""
    by = {r["arm"]: r for r in rows}
    fl = floors(by, refs, min_seeds)
    nat = {r["arm"]: r for r in (depth_native or [])}
    ref_names = set(refs.values())
    arms = [evaluate(r, fl, by, refs, min_seeds, nat.get(r["arm"]))
            for r in rows if r["arm"] not in ref_names]
    wbm = {m: _best([a for a in arms if a["model"] == m]) for m in MODELS}
    return {"floors": fl, "arms": arms, "winner_by_model": wbm, "winner_overall": _best(arms)}


def read_summary(path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    if rows and "arm" not in rows[0]:
        raise ValueError(f"{path}: no 'arm' column")
    return rows


def _fmt(x) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.4f}" if math.isfinite(x) else ("inf" if x > 0 else "-inf")
    return str(x)


def columns() -> list[str]:
    return (["arm", "model", "n_seeds"] + [f"{k}.mean" for k in TERMS]
            + [f"{k}.range" for k in TERMS] + [f"{k}.z" for k in TERMS]
            + ["eligible", "beyond_terms", "wins", "score", "class", "eligibility_row"]
            + [f"own_diff.{k}" for k in TERMS] + [f"own_beyond.{k}" for k in TERMS])


def _parse_ref(s: str):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--ref wants model=arm, got {s!r}")
    m, a = s.split("=", 1)
    return m.strip(), a.strip()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--summary", action="append", required=True)
    p.add_argument("--ref", action="append", type=_parse_ref, default=None)
    p.add_argument("--arm-prefix", default="")
    p.add_argument("--min-seeds", type=int, default=3)
    p.add_argument("--depth-native-summary", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    refs = dict(DEFAULT_REFS)
    if a.ref:
        refs.update(dict(a.ref))
    rows, seen = [], set()
    for path in a.summary:
        for r in read_summary(path):
            if r["arm"] in seen:
                print(f"ERROR: arm {r['arm']} appears in more than one summary row", file=sys.stderr)
                return 2
            seen.add(r["arm"])
            rows.append(r)
    ref_names = set(refs.values())
    rows = [r for r in rows if r["arm"] in ref_names or r["arm"].startswith(a.arm_prefix)]
    native = read_summary(a.depth_native_summary) if a.depth_native_summary else None
    try:
        res = select(rows, refs, a.min_seeds, depth_native=native)
    except RefError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    cols = columns()
    text = "\t".join(cols) + "\n" + "".join(
        "\t".join(_fmt(arm.get(c)) for c in cols) + "\n" for arm in res["arms"])
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    for k, f in res["floors"].items():
        print(f"# floor {k} = {f['best_mean']:.4f} - {f['best_range']:.4f} = {f['floor']:.4f}"
              f" (best base {f['best_model']})", file=sys.stderr)
    for m, w in res["winner_by_model"].items():
        print(f"# winner {m}: {w or 'none'}", file=sys.stderr)
    print(f"# winner overall: {res['winner_overall'] or 'none'}", file=sys.stderr)
    return 0 if res["winner_overall"] else 1


if __name__ == "__main__":
    sys.exit(main())
