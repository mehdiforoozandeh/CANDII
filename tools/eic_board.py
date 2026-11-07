#!/usr/bin/env python3
"""Build the landing-page EIC leaderboard: three layers of the 2019 challenge truth, CANDI on each.

The board shows the 2019 ENCODE Imputation Challenge field on the three truth versions the
organizers published scores for (Schreiber, Boix et al., Genome Biology 2023):

  original   the blind truth as scored in 2019                       Additional File 2 (MOESM2)
  qn         the same truth, quantile-normalized                      Additional File 3 (MOESM3)
  corrected  single-end reprocessed, then quantile-normalized         Additional File 4 (MOESM4)

Every number comes from Max's exact reproduction (his sprint 04), copied unchanged into
`leaderboard/eic/source/` (provenance in `source/SOURCE.md`). This tool scores nothing. It reads
those tables, refuses them if they stop describing a reproduced board, and writes `_site/`:

  python tools/eic_board.py check     gates + a double build diffed bit-exact
  python tools/eic_board.py build     writes _site/ (index.html, app.js, style.css, eic.json)

Protocol, where Max's code-verified reading and the paper's prose differ, follows Max: the 25
published entries keep their published scores and positions, and each CANDI arm is added alone
as a 26th entry and re-ranked (never two CANDI arms in one field). Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "leaderboard" / "eic"
DEFAULT_OUT = REPO / "_site"
SITE_FILES = ("index.html", "app.js", "style.css")
N_PUBLISHED = 25

# (layer id, board tsv, the `board` value of its rows in candi_placement.tsv)
LAYERS = (
    ("original", "pe_vs_MOESM2_board.tsv", "published MOESM2"),
    ("qn", "pq_vs_MOESM3_board.tsv", "published MOESM3"),
    ("corrected", "qn_vs_MOESM4_board.tsv", "published MOESM4"),
)

# CANDI arms on the board: (arm name in candi_placement.tsv, label on the page, shown by default)
ARMS = (
    ("CANDI + tcfloor", "CANDI + tcfloor", True),
    ("CANDI, uncorrected", "CANDI", True),
    ("CANDI repro-covariates + tcfloor", "CANDI + tcfloor, depth prompt 30M", False),
    ("CANDI repro-covariates, uncorrected", "CANDI, depth prompt 30M", False),
)

BASELINES = {"Average", "Avocado"}

# An exact reproduction: published and reproduced scores of MOESM2 and MOESM4 agree to float
# rounding. MOESM3 is a rebuild (median 2e-4 off; DNase 30%), so only its positions are gated.
EXACT_LAYERS = {"original", "corrected"}
EXACT_REL = 1e-8

# Loads the page may make. Plain links (<a href>) are fine; fetching anything is not.
EXTERNAL_LOAD = re.compile(r"""(src|@import|url\()\s*=?\s*["']?\s*https?://""", re.I)


class GateError(RuntimeError):
    pass


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def load_layer(root: Path, layer: str, tsv: str) -> List[Dict[str, Any]]:
    rows = read_tsv(root / "source" / tsv)
    if len(rows) != N_PUBLISHED:
        raise GateError(f"{tsv}: {len(rows)} entries, expected {N_PUBLISHED}")
    positions = sorted(int(r["position_pub"]) for r in rows)
    if positions != list(range(1, N_PUBLISHED + 1)):
        raise GateError(f"{tsv}: published positions are not 1..{N_PUBLISHED}")
    entries = []
    for r in rows:
        if r["position_ours"] != r["position_pub"] or r["shift"] != "0":
            raise GateError(f"{tsv}: {r['team']} is not in its published position — "
                            "this board is no longer a reproduction")
        pub, ours = float(r["score_mean_pub"]), float(r["score_mean_ours"])
        if layer in EXACT_LAYERS and abs(ours - pub) > EXACT_REL * max(abs(pub), 1e-12):
            raise GateError(f"{tsv}: {r['team']} reproduced {ours} vs published {pub}")
        entries.append({
            "name": r["team"],
            "kind": "baseline" if r["team"] in BASELINES else "team",
            "position": int(r["position_pub"]),
            "of": N_PUBLISHED,
            "score": round(pub, 4),
        })
    return sorted(entries, key=lambda e: e["position"])


def load_candi(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    rows = read_tsv(root / "source" / "candi_placement.tsv")
    out: Dict[str, List[Dict[str, Any]]] = {layer: [] for layer, _, _ in LAYERS}
    for layer, _, board in LAYERS:
        for arm, label, default in ARMS:
            hit = [r for r in rows if r["board"] == board and r["arm"] == arm]
            if len(hit) != 1:
                raise GateError(f"candi_placement.tsv: {len(hit)} rows for {arm!r} on {board}")
            r = hit[0]
            pos, of = int(r["position"]), int(r["of"])
            if of != N_PUBLISHED + 1 or not 1 <= pos <= of:
                raise GateError(f"{arm} on {board}: position {pos} of {of}")
            if int(r["n_experiments"]) != 51:
                raise GateError(f"{arm} on {board}: {r['n_experiments']} experiments, not 51")
            out[layer].append({
                "name": label,
                "kind": "candi",
                "default": default,
                "position": pos,
                "of": of,
                "score": round(float(r["score_mean"]), 4),
                "worst_bootstrap": round(float(r["score_worst_bootstrap"]), 4),
                "best_other": r["best_other"],
                "best_other_score": round(float(r["best_other_score"]), 4),
                "best_other_best_bootstrap": round(float(r["best_other_best_bootstrap"]), 4),
            })
    return out


def compile_board(root: Path) -> Dict[str, Any]:
    candi = load_candi(root)
    layers = []
    for layer, tsv, _ in LAYERS:
        layers.append({"id": layer, "published": load_layer(root, layer, tsv),
                       "candi": candi[layer]})
    return {"schema": "eic-board/1", "layers": layers}


def dump_json(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def cmd_build(args: argparse.Namespace) -> int:
    root, out = Path(args.root), Path(args.out)
    board = compile_board(root)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "eic.json").write_text(dump_json(board), encoding="utf-8")
    for name in SITE_FILES:
        src = root / "site" / name
        if not src.exists():
            raise GateError(f"site source {src} is missing")
        (out / name).write_bytes(src.read_bytes())
    print(f"built {out}: {len(board['layers'])} layers, "
          f"{sum(len(l['candi']) for l in board['layers'])} CANDI placements")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    root = Path(args.root)
    for name in SITE_FILES:
        src = root / "site" / name
        if not src.exists():
            raise GateError(f"site source {src} is missing")
        if EXTERNAL_LOAD.search(src.read_text(encoding="utf-8")):
            raise GateError(f"{src} loads an external resource — the page fetches nothing")
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        for target in (a, b):
            cmd_build(argparse.Namespace(root=root, out=target))
        for name in ("eic.json", *SITE_FILES):
            if (a / name).read_bytes() != (b / name).read_bytes():
                raise GateError(f"build is not deterministic: {name} differs between reruns")
    print("check passed: three reproduced boards, every CANDI placement in range, "
          "deterministic rebuild bit-exact")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eic_board.py", description=__doc__.splitlines()[0])
    p.add_argument("--root", default=str(ROOT), help="board directory (source/, site/)")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write _site/")
    b.add_argument("--out", default=str(DEFAULT_OUT))
    b.set_defaults(func=cmd_build)
    c = sub.add_parser("check", help="gates + a double build diffed bit-exact")
    c.set_defaults(func=cmd_check)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
