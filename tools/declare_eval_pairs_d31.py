#!/usr/bin/env python3
"""Write and re-check a regime's `eval_pairs` — the DECLARED imputation pairing (D31, t80).

    python tools/declare_eval_pairs.py declare --regime configs/regime.equiv.json \
        --input-prefix T_ --target-prefix V_ --target-prefix B_ \
        --out cruxvault/results/t80/eval_pairs.eic.json --regime-out /tmp/regime.paired.json

    python tools/declare_eval_pairs.py check --regime /tmp/regime.paired.json

`plan/BENCHMARK_DESIGN.md` §14 names this file as the thing that owns the pairing. It did not
exist, no shipped config declared `eval_pairs`, and `StoreSource` self-paired as a result — CANDI
sat a different exam from every rival on the board. This is the other half of that fix.

**Why a tool and not a library rule.** D16 makes store biosample names opaque ids that nothing may
parse, and D31 says the pairing is DECLARED, never inferred. Both would be violated by a
`T_X -> V_X` rule living in `candi.store` or `candi.bench`. They are not violated by an operator
running a naming rule ONCE, over one named corpus, and reading the result before it is committed:
the 38 EIC pairings are a claim about that corpus, and this file's job is to make the claim
checkable rather than to make it silently.

So the rule is an ARGUMENT, never a default. `--input-prefix` / `--target-prefix` say what the
convention is on this corpus; `--pairs-from` takes a two-column CSV instead and does no string
surgery at all. Neither is baked in, and `declare` refuses to run without one of them.

**What `check` is for.** `declare` writes a claim; `check` re-derives it from the store months
later and fails loudly if the corpus moved under it. It is the same three questions the design
asks of the panel — do the names exist, are the splits disjoint on `(cell, assay)`, and how many
experiments does each pair pose — answered against `manifest.json` rather than against a memo.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from candi.store import layout as L                                   # noqa: E402
from candi.store.reader import CorpusStore                            # noqa: E402
from candi.store.regime import Regime                                 # noqa: E402

Pairing = List[Tuple[str, str]]


# ---------------------------------------------------------------------------------------------
# where the pairing comes from
# ---------------------------------------------------------------------------------------------

def pairs_by_prefix(names: Sequence[str], input_prefix: str,
                    target_prefixes: Sequence[str]) -> Tuple[Pairing, List[str]]:
    """`(pairs, unpaired targets)` under the operator's naming rule.

    A target `<tp><cell>` pairs with `<input_prefix><cell>` when that name is in the corpus. A
    target whose partner is absent is RETURNED, not dropped: an unpaired truth cell is a hole in
    the panel, and the caller decides whether it is acceptable. Order is the corpus's own listing
    order, so two runs of this tool over one store emit byte-identical lists.
    """
    have = set(names)
    pairs: Pairing = []
    orphans: List[str] = []
    for name in names:
        tp = next((p for p in target_prefixes if name.startswith(p)), None)
        if tp is None:
            continue
        partner = input_prefix + name[len(tp):]
        if partner in have:
            pairs.append((partner, name))
        else:
            orphans.append(name)
    return pairs, orphans


def pairs_from_csv(path: Path) -> Pairing:
    """A two-column `input,target` CSV — the no-surgery source. `#` comments and a header are OK."""
    out: Pairing = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [c.strip() for c in line.split(",")]
        if len(parts) != 2 or not all(parts):
            raise SystemExit(f"{path}:{i}: expected `input,target`, got {line!r}")
        if parts[0].lower() == "input" and parts[1].lower() == "target":
            continue
        out.append((parts[0], parts[1]))
    if not out:
        raise SystemExit(f"{path}: no pairs")
    return out


# ---------------------------------------------------------------------------------------------
# what the corpus says about a pairing
# ---------------------------------------------------------------------------------------------

def panels(corpus: CorpusStore, pairs: Pairing) -> Dict[str, Dict[str, object]]:
    """Per pair: the input's assays, the target's, and the panel the harness will score.

    The panel rule is `StoreSource.targets`': *assays the truth cell has and the input cell does
    not* (`BENCHMARK_DESIGN.md` §5.1, §8). It is recomputed here from the corpus rather than
    imported, so the tool and the harness are two independent readings of one claim; a test pins
    them equal.
    """
    out: Dict[str, Dict[str, object]] = {}
    for a, b in pairs:
        xs = sorted(set(corpus[a].assays("counts")) - {L.CONTROL_TRACK})
        ts = sorted(set(corpus[b].assays("counts")) - {L.CONTROL_TRACK})
        out[f"{a}->{b}"] = {
            "input_assays": xs,
            "target_assays": ts,
            "panel": sorted(set(ts) - set(xs)),
            "overlap": sorted(set(ts) & set(xs)),
        }
    return out


def audit(corpus: CorpusStore, pairs: Pairing, *, orphans: Sequence[str] = ()
          ) -> Dict[str, object]:
    """Everything a reader needs to accept or reject the pairing, as plain JSON."""
    absent = sorted({n for p in pairs for n in p if n not in corpus})
    if absent:
        raise SystemExit(f"biosample(s) {absent} are not in {corpus.root}")
    per = panels(corpus, pairs)
    by_target: Dict[str, Dict[str, int]] = {}
    for (a, b), rec in zip(pairs, per.values()):
        head = b.split("_", 1)[0] + "_" if "_" in b else b
        acc = by_target.setdefault(head, {"cells": 0, "experiments": 0})
        acc["cells"] += 1
        acc["experiments"] += len(rec["panel"])
    return {
        "eval_pairs": [list(p) for p in pairs],
        "panels": per,
        "summary": {
            "pairs": len(pairs),
            "experiments": sum(len(r["panel"]) for r in per.values()),
            "by_target_prefix": by_target,
        },
        "disjoint": all(not r["overlap"] for r in per.values()),
        "overlaps": {k: r["overlap"] for k, r in per.items() if r["overlap"]},
        "empty_panels": [k for k, r in per.items() if not r["panel"]],
        "unpaired_targets": list(orphans),
        # The number the self-paired path could actually pose: a leave-one-out inside the truth
        # cell needs a second assay to prompt from. 16 of the 26 EIC `V_` cells fail that test,
        # which is the 45 -> 29 collapse t80 was opened for. Recorded so the fix stays legible.
        "self_paired_would_score": sum(len(r["target_assays"]) for r in per.values()
                                       if len(r["target_assays"]) > 1),
    }


def provenance(corpus: CorpusStore, rule: Dict[str, object]) -> Dict[str, object]:
    manifest = L.manifest_path(corpus.root)
    sha = (hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None)
    return {
        "tool": "tools/declare_eval_pairs.py",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "store": str(corpus.root),
        "manifest_sha256": sha,
        "rule": rule,
    }


def report(rec: Dict[str, object]) -> None:
    s = rec["summary"]
    print(f"pairs                {s['pairs']}")
    print(f"experiments          {s['experiments']}")
    for head, acc in sorted(s["by_target_prefix"].items()):
        print(f"  {head:<18} {acc['cells']} cell(s), {acc['experiments']} experiment(s)")
    print(f"disjoint             {rec['disjoint']}")
    if rec["overlaps"]:
        for k, v in sorted(rec["overlaps"].items()):
            print(f"  OVERLAP {k}: {v}  (not scored)")
    if rec["empty_panels"]:
        print(f"  EMPTY PANEL        {rec['empty_panels']}")
    if rec["unpaired_targets"]:
        print(f"  UNPAIRED TARGETS   {rec['unpaired_targets']}")
    print(f"self-paired would score  {rec['self_paired_would_score']} "
          f"of {s['experiments']} (the pre-t80 exam)")


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def _open(regime_path: Optional[str], store: Optional[str]) -> Tuple[CorpusStore, Optional[Regime]]:
    if (regime_path is None) == (store is None):
        raise SystemExit("exactly one of --regime / --store is required")
    if store is not None:
        return CorpusStore(Path(store)), None
    reg = Regime.from_file(Path(regime_path))
    return CorpusStore(Path(reg.store)), reg


def cmd_declare(a: argparse.Namespace) -> int:
    corpus, reg = _open(a.regime, a.store)
    if a.pairs_from:
        pairs, orphans = pairs_from_csv(Path(a.pairs_from)), []
        rule: Dict[str, object] = {"kind": "csv", "path": str(a.pairs_from)}
    elif a.input_prefix and a.target_prefix:
        pairs, orphans = pairs_by_prefix(list(corpus.biosamples), a.input_prefix, a.target_prefix)
        rule = {"kind": "prefix", "input_prefix": a.input_prefix,
                "target_prefixes": list(a.target_prefix)}
    else:
        raise SystemExit(
            "no pairing rule. Give --pairs-from <csv>, or --input-prefix with one or more "
            "--target-prefix. There is no default: D31 says the pairing is declared, and a "
            "default would make it inferred."
        )
    if not pairs:
        raise SystemExit(f"the rule matched no pair in {corpus.root}")
    rec = {**provenance(corpus, rule), **audit(corpus, pairs, orphans=orphans)}
    report(rec)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {a.out}")
    if a.regime_out:
        if reg is None:
            raise SystemExit("--regime-out needs --regime to copy from")
        obj = json.loads(Path(a.regime).read_text(encoding="utf-8"))
        obj["eval_pairs"] = [list(p) for p in pairs]
        Path(a.regime_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.regime_out).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
        Regime.from_file(Path(a.regime_out)).validate_against(corpus)   # it must still load
        print(f"wrote {a.regime_out}")
    return 0 if (rec["disjoint"] or a.allow_overlap) and not rec["empty_panels"] else 1


def cmd_check(a: argparse.Namespace) -> int:
    reg = Regime.from_file(Path(a.regime))
    if not reg.has_eval_pairs:
        print(f"{a.regime} declares no `eval_pairs`. The store path will SELF-PAIR: every eval "
              f"cell prompts itself, a cell holding one assay scores nothing, and the panel is "
              f"not the benchmark's. Run `declare`.")
        return 1
    corpus = CorpusStore(Path(reg.store))
    reg.validate_against(corpus)
    rec = audit(corpus, [(a_, b_) for a_, b_ in reg.eval_pairs])
    report(rec)
    return 0 if (rec["disjoint"] or a.allow_overlap) and not rec["empty_panels"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("declare", help="derive a pairing and write it down")
    d.add_argument("--regime", default=None, help="a regime file; its `store` key names the corpus")
    d.add_argument("--store", default=None, help="a CANDI_STORE root, instead of --regime")
    d.add_argument("--input-prefix", default=None, help="e.g. T_ — the prompt cells' prefix")
    d.add_argument("--target-prefix", action="append", default=[],
                   help="e.g. V_ (repeatable) — the truth cells' prefixes")
    d.add_argument("--pairs-from", default=None, help="a two-column input,target CSV instead")
    d.add_argument("--out", default=None, help="write the pairing + its provenance here")
    d.add_argument("--regime-out", default=None, help="write --regime with `eval_pairs` filled in")
    d.add_argument("--allow-overlap", action="store_true",
                   help="exit 0 even when a pair's two cells share an assay")
    d.set_defaults(fn=cmd_declare)

    c = sub.add_parser("check", help="re-derive a declared pairing from the store")
    c.add_argument("--regime", required=True)
    c.add_argument("--allow-overlap", action="store_true")
    c.set_defaults(fn=cmd_check)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":                                            # pragma: no cover
    raise SystemExit(main())
