"""t125 part 1 — rungs, the 432-row task table and the depth-only law pairs; pure functions.

**Rungs.** Row 1 A..D (g(C, C')), row 2 A2..D2 (g(eta_in, C, C')), and the old-spread A arm
`Aold` (row 1, t118 form A: one spread per pair). `FORM_MODULE` names each rung's f-form module.

**Tasks.** 432 = 8 rungs x G_IDS x MODELS x SEEDS (space counts only), rung slowest, seed fastest:
`index = rung_i*48 + g_i*6 + model_i*3 + seed` over RUNGS_ROW1 + RUNGS_ROW2 (0..383), then Aold:
`index = 384 + g_i*6 + model_i*3 + seed`; `run_name = f"{rung}_{g}_counts_{model}_s{seed}"`.

**Pairs.** `train_pairs` / `fit_pids` are t118's. `law_pairs_depth(rows, g)` = t118's law pairs
with `arm_src == arm_tgt == "depth"` (6 per track, 42 for "all"; PI decision (i));
`law_pairs(rows, g, scope="all")` = every t118 law pair.

    python tools/t125/eta/pairs.py tasks <MANIFEST.tsv>
    python tools/t125/eta/pairs.py count <MANIFEST.tsv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[2]
for _p in (_TOOLS / "t125", _TOOLS / "t118"):   # import convention: tools/t118 and tools/t125
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import pairs as _lp  # noqa: E402

RUNGS_ROW1 = ("A", "B", "C", "D")
RUNGS_ROW2 = ("A2", "B2", "C2", "D2")
RUNG_OLDSPREAD = "Aold"
RUNGS = RUNGS_ROW1 + RUNGS_ROW2 + (RUNG_OLDSPREAD,)
ROW1_OF = {"A2": "A", "B2": "B", "C2": "C", "D2": "D"}
MODELS = ("real", "nocov")
SEEDS = (0, 1, 2)
SPACE = "counts"
G_IDS = _lp.G_IDS
FORM_MODULE = {
    "A": "eta.fforms.form_a", "Aold": "ladder.fforms.form_a",
    "B": "ladder.fforms.form_b", "C": "ladder.fforms.form_c", "D": "ladder.fforms.form_d",
    "A2": "ladder.fforms.form_a2", "B2": "ladder.fforms.form_b2",
    "C2": "ladder.fforms.form_c2", "D2": "ladder.fforms.form_d2",
}
TASK_KEYS = ("index", "rung", "g", "model", "seed", "run_name")
LAW_SCOPES = ("depth", "all")

read_manifest = _lp.read_manifest
train_pairs = _lp.train_pairs
fit_pids = _lp.fit_pids


def row_of(rung: str) -> int:
    """1 for A..D and Aold, 2 for A2..D2; ValueError otherwise."""
    if rung in RUNGS_ROW1 or rung == RUNG_OLDSPREAD:
        return 1
    if rung in RUNGS_ROW2:
        return 2
    raise ValueError(f"rung {rung!r} not in {RUNGS}")


def run_name(rung: str, g: str, model: str, seed: int) -> str:
    return f"{rung}_{g}_{SPACE}_{model}_s{seed}"


def _task(index: int, rung: str, g: str, model: str, seed: int) -> dict:
    return {"index": index, "rung": rung, "g": g, "model": model, "seed": seed,
            "run_name": run_name(rung, g, model, seed)}


def tasks(rows: list[dict] | None = None) -> list[dict]:
    """The 432-row task table; a pure function of the constants (rows kept for the signature)."""
    out = []
    for ri, rung in enumerate(RUNGS_ROW1 + RUNGS_ROW2):
        for gi, g in enumerate(G_IDS):
            for mi, model in enumerate(MODELS):
                for seed in SEEDS:
                    out.append(_task(ri * 48 + gi * 6 + mi * 3 + seed, rung, g, model, seed))
    for gi, g in enumerate(G_IDS):
        for mi, model in enumerate(MODELS):
            for seed in SEEDS:
                out.append(_task(384 + gi * 6 + mi * 3 + seed, RUNG_OLDSPREAD, g, model, seed))
    assert [t["index"] for t in out] == list(range(len(out)))
    return out


def f_runs_seed0() -> list[str]:
    """The 64 real seed-0 run names of rows 1 and 2 (Aold excluded), in task order."""
    return [t["run_name"] for t in tasks()
            if t["model"] == "real" and t["seed"] == 0 and t["rung"] != RUNG_OLDSPREAD]


def law_pairs_depth(rows: list[dict], g: str) -> list[dict]:
    """t118's law pairs with arm_src == arm_tgt == "depth" (6 per track, 42 for "all")."""
    return [p for p in _lp.law_pairs(rows, g)
            if p["arm_src"] == "depth" and p["arm_tgt"] == "depth"]


def law_pairs(rows: list[dict], g: str, scope: str = "depth") -> list[dict]:
    """scope "depth": `law_pairs_depth`; scope "all": every t118 arm->arm law pair."""
    if scope == "depth":
        return law_pairs_depth(rows, g)
    if scope == "all":
        return _lp.law_pairs(rows, g)
    raise ValueError(f"scope {scope!r} not in {LAW_SCOPES}")


def count_lines(rows: list[dict]) -> list[str]:
    return [f"tasks {len(tasks(rows))}", f"f_runs_seed0 {len(f_runs_seed0())}",
            f"law_pairs_depth all {len(law_pairs_depth(rows, 'all'))}"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tasks", "count"):
        sub.add_parser(name).add_argument("manifest")
    args = ap.parse_args(argv)
    rows = read_manifest(args.manifest)
    if args.cmd == "count":
        sys.stdout.write("".join(line + "\n" for line in count_lines(rows)))
    else:
        sys.stdout.write("\t".join(TASK_KEYS) + "\n" + "".join(
            "\t".join(str(t[k]) for k in TASK_KEYS) + "\n" for t in tasks(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
