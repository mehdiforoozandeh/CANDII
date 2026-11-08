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

# The error decomposition (decomp/summary.tsv, Max's sprint 06 method): its `truth` code per layer.
DECOMP_TRUTH = {"original": "pe", "qn": "pq", "corrected": "qn"}
N_EXPERIMENTS = 51
# Entries with fewer scored tracks than experiments: UIOWA submitted nothing for one experiment.
SHORT_ENTRIES = {"UIOWA Michaelson Lab": 50}
# The tables are printed to 6 significant figures, so every sum below holds to ~1e-5 of max(mse, 1).
DECOMP_REL = 1e-4

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


BANDS = ("fine", "peak", "domain", "broad")   # Haar bands: <=200 bp, 0.4-3.2 kb, 6.4-51 kb, >=102 kb
PARTS = ("shared", "specific")


def _near(a: float, b: float, scale: float) -> bool:
    return abs(a - b) <= DECOMP_REL * max(scale, 1.0)


def load_decomp(root: Path, layer: str, names: Sequence[str]) -> Dict[str, Any]:
    """The error budget of every board entry on one layer, from decomp/summary.tsv and
    decomp/components.tsv (rows with assay == "all").

    Unit: normalized MSE, the squared error in asinh units divided by the truth track's variance
    (1.0 = predicting each track's own mean), averaged over the scored tracks. Per entry it splits
    exactly into offset + scale + nonlinear (calibration) + shared and specific error in four Haar
    bands; per band and part, the truth's variance V = explained + unexplained, and the error
    E = unexplained + added. Returns {"truth": the truth's variance split, "entries": [...]}."""
    truth = DECOMP_TRUTH[layer]
    labels = {arm: label for arm, label, _ in ARMS}
    rows = [r for r in read_tsv(root / "decomp" / "summary.tsv")
            if r["truth"] == truth and r["assay"] == "all"]
    by_name = {labels.get(r["entry"], r["entry"]): r for r in rows}
    if len(by_name) != len(rows) or sorted(by_name) != sorted(names):
        raise GateError(f"decomp/summary.tsv ({truth}): entries do not match the board: "
                        f"missing {sorted(set(names) - set(by_name))}, "
                        f"extra {sorted(set(by_name) - set(names))}")
    comps: Dict[str, Dict[tuple, Dict[str, float]]] = {}
    for c in read_tsv(root / "decomp" / "components.tsv"):
        if c["truth"] == truth and c["assay"] == "all":
            comps.setdefault(labels.get(c["entry"], c["entry"]), {})[(c["part"], c["band"])] = {
                k: float(c[k]) for k in ("V", "explained", "unexplained", "added", "E")}
    truth_split = None
    out = []
    for name in names:
        r = by_name[name]
        n = int(r["n_tracks"])
        if n != SHORT_ENTRIES.get(r["entry"], N_EXPERIMENTS):
            raise GateError(f"decomp ({truth}) {name}: {n} tracks")
        f = {k: float(v) for k, v in r.items() if k not in ("truth", "entry", "assay", "n_tracks")}
        mse = f["mse"]
        if not _near(mse, f["level"] + f["shared"] + f["specific"], mse):
            raise GateError(f"decomp ({truth}) {name}: mse {mse} != level + shared + specific")
        if not _near(f["level"], f["offset"] + f["scale"] + f["nonlinear"], mse):
            raise GateError(f"decomp ({truth}) {name}: level != offset + scale + nonlinear")
        for part, pre in (("shared", "sh"), ("specific", "sp")):
            if not _near(f[part], sum(f[f"{pre}_{b}"] for b in BANDS), mse):
                raise GateError(f"decomp ({truth}) {name}: {part} != the sum of its bands")
        cs = comps.get(name, {})
        if sorted(cs) != sorted([(p, b) for p in PARTS for b in BANDS] + [("level", "all")]):
            raise GateError(f"decomp/components.tsv ({truth}) {name}: missing components")
        expl, unexpl, added = [], [], 0.0
        for part, pre in (("shared", "sh"), ("specific", "sp")):
            for b in BANDS:
                c = cs[(part, b)]
                if not (_near(c["V"], c["explained"] + c["unexplained"], mse)
                        and _near(c["E"], c["unexplained"] + c["added"], mse)
                        and _near(c["E"], f[f"{pre}_{b}"], mse)
                        and _near(c["V"], f[f"truth_{pre}_{b}"], mse)):
                    raise GateError(f"decomp ({truth}) {name}: {part} {b} does not add up")
                expl.append(round(c["explained"], 4))
                unexpl.append(round(c["unexplained"], 4))
                added += c["added"]
        if not _near(cs[("level", "all")]["E"], f["level"], mse):
            raise GateError(f"decomp ({truth}) {name}: the level component != level")
        tv = [f[f"truth_{pre}_{b}"] for pre in ("sh", "sp") for b in BANDS]
        if n == N_EXPERIMENTS:   # the truth's split is the same for every entry with all tracks
            if truth_split is None:
                if abs(sum(tv) - 1.0) > 1e-3:
                    raise GateError(f"decomp ({truth}): the truth's variance split sums to {sum(tv)}")
                truth_split = tv
            elif any(abs(a - b) > DECOMP_REL for a, b in zip(tv, truth_split)):
                raise GateError(f"decomp ({truth}) {name}: the truth's variance split differs")
        out.append({"name": name, "n_tracks": n, "mse": round(mse, 4), "level": round(f["level"], 4),
                    "offset": round(f["offset"], 4), "scale": round(f["scale"], 4),
                    "nonlinear": round(f["nonlinear"], 4),
                    "shared": round(f["shared"], 4), "specific": round(f["specific"], 4),
                    "bands": [round(f[f"{pre}_{b}"], 4) for pre in ("sh", "sp") for b in BANDS],
                    "explained": expl, "unexplained": unexpl, "added": round(added, 4)})
    return {"truth": [round(v, 4) for v in truth_split], "entries": out}


def compile_board(root: Path) -> Dict[str, Any]:
    candi = load_candi(root)
    layers = []
    for layer, tsv, _ in LAYERS:
        published = load_layer(root, layer, tsv)
        default_arms = [c["name"] for c in candi[layer] if c["default"]]
        dec = load_decomp(root, layer, [p["name"] for p in published] + default_arms)
        layers.append({"id": layer, "published": published, "candi": candi[layer],
                       "decomp": dec["entries"], "decomp_truth": dec["truth"]})
    return {"schema": "eic-board/3", "layers": layers}


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
