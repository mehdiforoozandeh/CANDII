"""The h task tables: formula h, learned h on the true X', learned h on f's NB, alone and chain lists.

H_FORMULA  the two formula h (nothing learned): bin-only and fragment-window.
H_LEARNED  12 (variant, rung, noise): affine and mlp at rungs 1 and 5, formula_affine and
           formula_mlp at rung 5 only, each push only (noise 0) and push + own noise (noise 1).

tasks_true()        36 rows = H_LEARNED x seeds 0..2, index = learned_i*3 + seed, source "true".
tasks_fnb(f_runs)   f_runs x H_LEARNED x seeds (768 with 64 f runs and one seed),
                    index = f_i*(12*len(seeds)) + learned_i*len(seeds) + seed_i,
                    source = "fnb:<f_run>".
tasks_alone()       38 rows: the 2 formulas, then the 36 true-trained h (h scored on the true X').
chain_h_list(f, s)  the h scored in the chain on f: 2 formulas + 36 true h + this f's 12*s fnb h.

    python tools/t125/hmap/tasks.py true  <MANIFEST.tsv>
    python tools/t125/hmap/tasks.py fnb   <MANIFEST.tsv> [--fnb-seeds 1|3]   (f runs: eta.pairs.f_runs_seed0)
    python tools/t125/hmap/tasks.py alone <MANIFEST.tsv>
    python tools/t125/hmap/tasks.py chain-list <f_run> [--fnb-seeds 1|3]     (one h name per line)

The manifest is read for the signature only (as `eta/pairs.py tasks`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_T125 = Path(__file__).resolve().parents[1]
for _p in (_T125.parent / "t118", _T125):          # script mode: ladder, eta and hmap importable
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

H_FORMULA = (("formula", "bin"), ("formula", "frag"))
H_LEARNED = ([(variant, rung, noise) for variant in ("affine", "mlp") for rung in (1, 5)
              for noise in (0, 1)]
             + [(variant, 5, noise) for variant in ("formula_affine", "formula_mlp")
                for noise in (0, 1)])
TRUE_SEEDS = (0, 1, 2)
TASK_KEYS = ("index", "variant", "rung", "noise", "source", "seed", "h_name")


def h_name(variant: str, rung: int, noise: int, source_tag: str, seed: int) -> str:
    return f"h_{variant}_r{rung}_{'noise' if noise else 'push'}_{source_tag}_s{seed}"


def formula_name(window: str) -> str:
    return f"h_formula_{window}"


def _seeds(seeds) -> tuple[int, ...]:
    """An int N means seeds 0..N-1; a sequence is taken as given."""
    if isinstance(seeds, int):
        return tuple(range(seeds))
    return tuple(int(s) for s in seeds)


def tasks_true() -> list[dict]:
    rows = []
    for li, (variant, rung, noise) in enumerate(H_LEARNED):
        for seed in TRUE_SEEDS:
            rows.append({"index": li * len(TRUE_SEEDS) + seed, "variant": variant, "rung": rung,
                         "noise": noise, "source": "true", "seed": seed,
                         "h_name": h_name(variant, rung, noise, "true", seed)})
    return rows


def tasks_fnb(f_runs: list[str], seeds=(0,)) -> list[dict]:
    seeds = _seeds(seeds)
    per_f = len(H_LEARNED) * len(seeds)
    rows = []
    for fi, f_run in enumerate(f_runs):
        for li, (variant, rung, noise) in enumerate(H_LEARNED):
            for si, seed in enumerate(seeds):
                rows.append({"index": fi * per_f + li * len(seeds) + si, "variant": variant,
                             "rung": rung, "noise": noise, "source": f"fnb:{f_run}",
                             "seed": seed, "h_name": h_name(variant, rung, noise, f_run, seed)})
    return rows


def tasks_alone() -> list[dict]:
    rows = [{"index": i, "variant": variant, "rung": "", "noise": "", "source": f"formula:{window}",
             "seed": "", "h_name": formula_name(window)}
            for i, (variant, window) in enumerate(H_FORMULA)]
    off = len(rows)
    for r in tasks_true():
        rows.append(dict(r, index=r["index"] + off))
    return rows


def chain_h_list(f_run: str, fnb_seeds=1) -> list[str]:
    return ([formula_name(w) for _, w in H_FORMULA]
            + [r["h_name"] for r in tasks_true()]
            + [r["h_name"] for r in tasks_fnb([f_run], seeds=fnb_seeds)])


def _tsv(rows: list[dict]) -> str:
    lines = ["\t".join(TASK_KEYS)]
    lines += ["\t".join(str(r[k]) for k in TASK_KEYS) for r in rows]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("true", "fnb", "alone"):
        p = sub.add_parser(name)
        p.add_argument("manifest")
        if name == "fnb":
            p.add_argument("--fnb-seeds", type=int, choices=(1, 3), default=1)
    p = sub.add_parser("chain-list")
    p.add_argument("f_run")
    p.add_argument("--fnb-seeds", type=int, choices=(1, 3), default=1)
    args = ap.parse_args(argv)

    if args.cmd == "chain-list":
        sys.stdout.write("\n".join(chain_h_list(args.f_run, args.fnb_seeds)) + "\n")
        return 0
    from ladder import pairs as lpairs
    lpairs.read_manifest(args.manifest)            # signature only
    if args.cmd == "true":
        rows = tasks_true()
    elif args.cmd == "alone":
        rows = tasks_alone()
    else:
        from eta.pairs import f_runs_seed0         # chunk t125-P1a
        rows = tasks_fnb(f_runs_seed0(), seeds=args.fnb_seeds)
    sys.stdout.write(_tsv(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
