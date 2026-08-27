"""The rivals leaderboard compiler (LEADERBOARD_PRD.md, t58). Three verbs:

    python tools/leaderboard.py add <score.json> --board main --method Avocado --version 2026-08-27 \\
        --date 2026-08-27 --lineage rival --position-class transductive --cell-class retrained \\
        --scoring-sha <sha> --store-manifest-hash <hash>
    python tools/leaderboard.py build
    python tools/leaderboard.py check

`add` stamps one `candi.bench.external`-shaped score json into one row file under
`leaderboard/rows/<board>/<method>@<version>.json`. It computes nothing: every number in a row is
copied from the score file's macro block, gated on the way in — NaN refused, provenance mandatory,
the board's frozen eval-set hash enforced, and the companion rules of the registry (count `crps`
never without `crps_oracle_scaled` + `scale_error`; pval `crps` never without `pit_ks` +
`coverage_95`) applied as refusals rather than as conventions.

`build` compiles registry + boards + rows into `_site/leaderboard.json` and copies the static site
beside it. It is a pure function of the repo tree: sorted keys, no timestamps, bit-identical reruns.

`check` re-runs every gate on the existing rows, rebuilds twice, and diffs the two builds bit-exact.
CI runs it before every deploy.

The composite (PRD §5.4): per in-composite metric, rows whose gap is under the metric's noise floor
are tied and share a rank interval; category sub-score = mean rank interval over the category's
ranked metrics; composite = unweighted mean of category intervals; the displayed rank is the spread
of best and worst achievable rank. A category enters the composite only when every ranked row on the
board has all of its ranked metrics (§5.2) — the count arm therefore ranks on its own sub-board.

Stdlib only, by rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO / "leaderboard"
DEFAULT_OUT = REPO / "_site"

SLUG = re.compile(r"^[A-Za-z0-9._-]+$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TODO_HASH = "TODO-"

#: provenance keys `add` copies into the row's flags-of-record block when the score json has them.
FLAG_KEYS = ("crps_estimator", "crps_k", "crps_seed", "allow_missing", "msevar",
             "signal_target_transform", "pval_pred_space", "pred_inversion", "seed",
             "placement_method", "aggregation", "n_experiments", "contributor_mode")

LINEAGES = ("candi", "rival", "baseline", "entrant")
POSITION_CLASSES = {"transductive": "position-transductive", "generalizing": "position-generalizing",
                    "unrecorded": "position class unrecorded"}
CELL_CLASSES = {"zero-shot": "zero-shot cell types", "retrained": "retrained per setting",
                "unrecorded": "cell-type class unrecorded"}

#: the scorer name stamped on Dataset-3 placement rows (t54); `check` uses it to know how to
#: re-extract such a row from its artifact.
PLACEMENT_SCORER = "vendored-001-scorer (t54)"


class GateError(SystemExit):
    """A refused row or a failed check. Message first, exit code 1."""

    def __init__(self, msg: str) -> None:
        super().__init__(f"leaderboard: {msg}")


def _refuse_nan(token: str) -> float:
    raise GateError(f"score json carries a non-finite literal ({token}); NaN never enters a row")


def load_json(path: Path, allow_nonfinite: bool = False) -> Any:
    """allow_nonfinite is for t54 placement files only: their per_assay blocks carry NaN for
    assay-specific metrics (prom_corr outside H3K4me3). Nothing non-finite can still enter a
    row — extract_metrics refuses it per value, and dump_json refuses it on the way out."""
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          parse_constant=None if allow_nonfinite else _refuse_nan)
    except FileNotFoundError:
        raise GateError(f"{path} does not exist")
    except json.JSONDecodeError as e:
        raise GateError(f"{path} is not valid JSON: {e}")


def dump_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


# ---------------------------------------------------------------- registry ---

def load_registry(root: Path) -> Dict[str, Any]:
    reg = load_json(root / "registry.json")
    for field in ("metrics", "categories", "candi_self_comparison_bar"):
        if field not in reg:
            raise GateError(f"registry.json is missing `{field}`")
    for m in reg["metrics"]:
        for field in ("key", "arm", "category", "direction", "role", "floor", "decimals"):
            if field not in m:
                raise GateError(f"registry metric {m.get('key')} is missing `{field}`")
        if m["category"] not in reg["categories"]:
            raise GateError(f"registry metric {m['key']} names unknown category {m['category']}")
        if m["role"] == "ranked" and m["direction"] not in ("higher", "lower"):
            raise GateError(f"ranked metric {m['arm']}/{m['key']} needs a direction")
    return reg


def load_boards(root: Path) -> Dict[str, Any]:
    boards = load_json(root / "boards.json")
    if "boards" not in boards:
        raise GateError("boards.json is missing `boards`")
    for bid, b in boards["boards"].items():
        for field in ("label", "protocol", "eval_set", "frozen", "views", "caveats"):
            if field not in b:
                raise GateError(f"board `{bid}` is missing `{field}`")
        for field in ("store_manifest_hash", "regime_sha256"):
            if field not in b["frozen"]:
                raise GateError(f"board `{bid}` frozen block is missing `{field}`")
    return boards


def metric_slot(metric: Mapping[str, Any]) -> str:
    """Which sub-dict of row['metrics'] a registry metric lives in."""
    return metric["arm"] if metric["arm"] in ("pval", "count") else "diagnostics"


def metric_id(metric: Mapping[str, Any]) -> str:
    return f"{metric_slot(metric)}/{metric['key']}"


# ---------------------------------------------------------------- add ---

def extract_metrics(score: Mapping[str, Any], registry: Mapping[str, Any],
                    allow_missing: bool) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Copy registry metrics out of a bench-shaped score json. Returns (metrics, missing ids)."""
    macro = score.get("macro")
    if not isinstance(macro, dict):
        raise GateError("score json has no `macro` block — not a candi.bench-shaped file")
    cblock = score.get("C") if isinstance(score.get("C"), dict) else {}
    out: Dict[str, Dict[str, float]] = {}
    missing: List[str] = []
    for m in registry["metrics"]:
        slot = metric_slot(m)
        if slot == "diagnostics":
            src = cblock
        else:
            src = macro.get(m["arm"], {}) if isinstance(macro.get(m["arm"]), dict) else {}
        val = src.get(m["key"])
        if val is None:
            # diagnostics are CANDI-lineage instruments; rivals lack them by construction and
            # their absence never needs --allow-missing.
            if slot != "diagnostics":
                missing.append(metric_id(m))
            continue
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
            raise GateError(f"metric {metric_id(m)} is not a finite number: {val!r}")
        out.setdefault(slot, {})[m["key"]] = float(val)
    if missing and not allow_missing:
        raise GateError("score json lacks registry metrics "
                        f"{', '.join(missing)}; re-run with --allow-missing to record the gap")
    check_companions(out, registry)
    return out, sorted(missing)


def check_companions(metrics: Mapping[str, Mapping[str, float]],
                     registry: Mapping[str, Any]) -> None:
    """A metric never enters without the companions the registry declares for it."""
    for m in registry["metrics"]:
        comps = m.get("companions") or []
        if not comps:
            continue
        slot = metric_slot(m)
        have = metrics.get(slot, {})
        if m["key"] in have:
            lacked = [c for c in comps if c not in have]
            if lacked:
                raise GateError(f"{metric_id(m)} present without its companion(s) "
                                f"{', '.join(lacked)} — refused (registry companion rule)")


def placement_score(placement: Mapping[str, Any], method: str, path: Path) -> Dict[str, Any]:
    """Shape one method of a t54 Dataset-3 placement.json like a bench score, so the same
    gates serve both. Numbers are copied from `macro_all`; nothing is computed."""
    methods = placement.get("methods")
    if not isinstance(methods, dict) or method not in methods:
        raise GateError(f"{path} has no method `{method}` "
                        f"({'not a placement file' if not isinstance(methods, dict) else 'check the spelling'})")
    return {
        "provenance": {
            "suite": PLACEMENT_SCORER,
            "regime": "Synapse syn17083203 blind-test tracks",
            "sigma_table": None,
            # Dataset-3 signal for the scored marks IS -log10 p (t54 caveat 2; DNase excluded)
            "pval_pred_space": "-log10p",
            "aggregation": placement.get("aggregation"),
            "n_experiments": methods[method].get("n_experiments"),
            "placement_method": method,
        },
        "macro": {"pval": methods[method]["macro_all"], "count": {}},
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_path_of_record(path: Path) -> str:
    """Store vault evidence paths repo-relative, so any checkout resolves them."""
    s = str(path.resolve())
    marker = "/cruxvault/results/"
    return "cruxvault/results/" + s.split(marker, 1)[1] if marker in s else str(path)


def build_provenance(score: Mapping[str, Any], args: argparse.Namespace,
                     score_path: Path) -> Dict[str, Any]:
    prov = score.get("provenance")
    if not isinstance(prov, dict):
        raise GateError("score json has no `provenance` block")
    if "suite" not in prov:
        raise GateError("score provenance names no `suite` (scorer)")
    fir_path = args.fir_path
    if fir_path is None:
        sibling = score_path.parent / "FIR_PATH.txt"
        if sibling.exists():
            fir_path = sibling.read_text(encoding="utf-8").strip().splitlines()[0]
    if not fir_path:
        raise GateError("no FIR path: pass --fir-path or keep FIR_PATH.txt beside the score json")
    sigma = prov.get("sigma_table")
    sigma_id = None
    if isinstance(sigma, dict):
        sigma_id = {"method": sigma.get("method"), "fitted_on": sigma.get("fitted_on")}
        if not sigma_id["method"] or not sigma_id["fitted_on"]:
            raise GateError("score provenance sigma_table lacks method/fitted_on — no σ-table id")
    flags = {k: prov[k] for k in FLAG_KEYS if k in prov}
    out = {
        "scoring_sha": args.scoring_sha,
        "score_json": score_path_of_record(score_path),
        "fir_path": fir_path,
        "store_manifest_hash": args.store_manifest_hash,
        "regime": prov.get("regime"),
        "sigma_table": sigma_id,
        "scorer": prov["suite"],
        "flags": flags,
        "artifacts_resolved_at_add": score_path.exists(),
    }
    # method-level caveats recorded by the producer travel on the row (e.g. eDICE's
    # transductive + masking-rate caveats in its prediction manifest)
    manifest = prov.get("manifest") if isinstance(prov.get("manifest"), dict) else {}
    notes = {k: manifest[k] for k in ("notes", "caveat", "masking_caveat") if manifest.get(k)}
    if notes:
        out["method_notes"] = notes
    return out


def gate_row_against_board(row: Mapping[str, Any], board: Mapping[str, Any], bid: str) -> None:
    frozen = board["frozen"]
    if str(frozen["store_manifest_hash"]).startswith(TODO_HASH):
        raise GateError(f"board `{bid}` eval-set hashes are not frozen yet; "
                        "stamp boards.json `frozen` before adding rows")
    prov = row["provenance"]
    if prov["store_manifest_hash"] != frozen["store_manifest_hash"]:
        raise GateError(f"row store manifest hash {prov['store_manifest_hash']} does not match "
                        f"board `{bid}` frozen hash {frozen['store_manifest_hash']} — "
                        "nobody quietly scores on easier data")
    board_regime = board["eval_set"].get("regime", "")
    if board_regime.endswith(".json"):
        row_regime = prov.get("regime") or ""
        if Path(row_regime).name != Path(board_regime).name:
            raise GateError(f"row regime `{row_regime}` is not board `{bid}`'s "
                            f"regime `{board_regime}`")
    if row["metrics"].get("pval") and prov["flags"].get("pval_pred_space") != "-log10p":
        raise GateError("pval metrics extracted but provenance lacks pval_pred_space=-log10p — "
                        "the row predates the spaces contract and is not quotable (EVAL.md)")


def gate_row_shape(row: Mapping[str, Any], registry: Mapping[str, Any]) -> None:
    for field in ("board", "method", "version", "date", "lineage", "badges", "metrics",
                  "missing_metrics", "provenance"):
        if field not in row:
            raise GateError(f"row {row.get('method')}@{row.get('version')} is missing `{field}`")
    if not SLUG.match(row["method"]) or not SLUG.match(row["version"]):
        raise GateError(f"row method/version must match {SLUG.pattern}")
    if not DATE.match(row["date"]):
        raise GateError(f"row date `{row['date']}` is not YYYY-MM-DD")
    if row["lineage"] not in LINEAGES:
        raise GateError(f"row lineage `{row['lineage']}` is not one of {LINEAGES}")
    prov = row["provenance"]
    for field in ("scoring_sha", "score_json", "fir_path", "store_manifest_hash",
                  "regime", "scorer", "flags"):
        if not prov.get(field) and prov.get(field) != {}:
            raise GateError(f"row {row['method']}@{row['version']} provenance lacks `{field}` — "
                            "no provenance, no row (PRD §7.2)")
    for metrics in (row["metrics"], row.get("metrics_strict") or {}):
        for slot, block in metrics.items():
            for key, val in block.items():
                if not isinstance(val, (int, float)) or isinstance(val, bool) \
                        or not math.isfinite(val):
                    raise GateError(f"row {row['method']}@{row['version']} metric {slot}/{key} "
                                    f"is not finite: {val!r}")
        check_companions(metrics, registry)


def cmd_add(args: argparse.Namespace) -> int:
    root = Path(args.root)
    registry = load_registry(root)
    boards = load_boards(root)
    if args.board not in boards["boards"]:
        raise GateError(f"unknown board `{args.board}`; boards.json has "
                        f"{sorted(boards['boards'])}")
    board = boards["boards"][args.board]
    score_path = Path(args.score)
    if args.placement_method:
        # Dataset-3 rows: the artifact itself is pinned by the board's frozen regime hash
        digest = sha256_file(score_path)
        if digest != board["frozen"]["regime_sha256"]:
            raise GateError(f"{score_path} digests to {digest[:12]}…, not the placement "
                            f"artifact board `{args.board}` froze — refused")
        score = placement_score(load_json(score_path, allow_nonfinite=True),
                                args.placement_method, score_path)
    else:
        score = load_json(score_path)
    metrics, missing = extract_metrics(score, registry, args.allow_missing)
    row: Dict[str, Any] = {
        "schema_version": 1,
        "board": args.board,
        "method": args.method,
        "version": args.version,
        "date": args.date,
        "lineage": args.lineage,
        "badges": {
            "position": POSITION_CLASSES[args.position_class],
            "cell_types": CELL_CLASSES[args.cell_class],
        },
        "metrics": metrics,
        "missing_metrics": missing,
        "provenance": build_provenance(score, args, score_path),
    }
    if args.strict_score:
        if "strict" not in board.get("views", []):
            raise GateError(f"board `{args.board}` has no strict view; --strict-score refused")
        strict_metrics, strict_missing = extract_metrics(load_json(Path(args.strict_score)),
                                                         registry, args.allow_missing)
        row["metrics_strict"] = strict_metrics
        row["missing_metrics_strict"] = strict_missing
        row["provenance"]["strict_score_json"] = str(args.strict_score)
    gate_row_shape(row, registry)
    gate_row_against_board(row, board, args.board)
    out = root / "rows" / args.board / f"{args.method}@{args.version}.json"
    if out.exists() and not args.force:
        raise GateError(f"{out} already exists; pass --force to restamp")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json(row), encoding="utf-8")
    print(f"stamped {out.relative_to(root)}")
    return 0


# ---------------------------------------------------------------- ranking ---

def rank_spreads(values: Mapping[str, float], direction: str,
                 floor: Optional[float]) -> Dict[str, Tuple[int, int]]:
    """PRD §5.4.1 — rank spread per row id. Two rows whose gap is under the floor are tied.

    best = 1 + #{j beats i by more than the floor}; worst = N − #{i beats j by more than it}.
    No floor → floor 0: only exact ties share a rank.
    """
    f = floor or 0.0
    sign = -1.0 if direction == "higher" else 1.0
    ids = sorted(values)
    n = len(ids)
    out: Dict[str, Tuple[int, int]] = {}
    for i in ids:
        vi = sign * values[i]
        beat_me = sum(1 for j in ids if j != i and vi - sign * values[j] > f)
        i_beat = sum(1 for j in ids if j != i and sign * values[j] - vi > f)
        out[i] = (1 + beat_me, n - i_beat)
    return out


def interval_rank_spreads(intervals: Mapping[str, Tuple[float, float]]) -> Dict[str, Tuple[int, int]]:
    """Rank spread over [lo, hi] score intervals, lower is better.

    j is certainly better than i when hi_j < lo_i; both spreads count only certainties.
    """
    ids = sorted(intervals)
    n = len(ids)
    out: Dict[str, Tuple[int, int]] = {}
    for i in ids:
        lo_i, hi_i = intervals[i]
        surely_above = sum(1 for j in ids if j != i and intervals[j][1] < lo_i)
        surely_below = sum(1 for j in ids if j != i and hi_i < intervals[j][0])
        out[i] = (1 + surely_above, n - surely_below)
    return out


def compile_view(rows: Sequence[Mapping[str, Any]], registry: Mapping[str, Any],
                 view: str) -> Dict[str, Any]:
    """One board view: ranks, category sub-scores, composite spreads, sub-boards."""
    metrics_field = "metrics_strict" if view == "strict" else "metrics"
    ranked = [r for r in rows if r.get(metrics_field)]
    unranked = [r for r in rows if not r.get(metrics_field)]
    by_id = {f"{r['method']}@{r['version']}": r for r in ranked}

    def val(rid: str, m: Mapping[str, Any]) -> Optional[float]:
        return by_id[rid][metrics_field].get(metric_slot(m), {}).get(m["key"])

    ranked_metrics = [m for m in registry["metrics"] if m["role"] == "ranked"]
    metric_ranks: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for m in ranked_metrics:
        have = {rid: val(rid, m) for rid in by_id}
        have = {rid: v for rid, v in have.items() if v is not None}
        if have:
            metric_ranks[metric_id(m)] = rank_spreads(have, m["direction"], m["floor"])

    # §5.2 — a category enters the composite only when every ranked row has all of its
    # ranked metrics.
    cats = registry["categories"]
    composite_cats = []
    for cid, cat in sorted(cats.items()):
        if not cat.get("in_composite"):
            continue
        cat_metrics = [m for m in ranked_metrics if m["category"] == cid]
        if not cat_metrics:
            continue
        if all(val(rid, m) is not None for rid in by_id for m in cat_metrics):
            composite_cats.append(cid)

    out_rows = []
    intervals: Dict[str, Tuple[float, float]] = {}
    for rid, r in by_id.items():
        subscores: Dict[str, Tuple[float, float]] = {}
        for cid in composite_cats:
            spreads = [metric_ranks[metric_id(m)][rid]
                       for m in ranked_metrics if m["category"] == cid]
            subscores[cid] = (sum(s[0] for s in spreads) / len(spreads),
                              sum(s[1] for s in spreads) / len(spreads))
        composite = None
        if subscores:
            composite = (sum(s[0] for s in subscores.values()) / len(subscores),
                         sum(s[1] for s in subscores.values()) / len(subscores))
            intervals[rid] = composite
        out_rows.append({
            "id": rid,
            "method": r["method"],
            "version": r["version"],
            "date": r["date"],
            "lineage": r["lineage"],
            "badges": r["badges"],
            "metrics": r[metrics_field],
            "missing_metrics": r.get("missing_metrics_strict" if view == "strict"
                                     else "missing_metrics", []),
            "provenance": r["provenance"],
            "verified": bool(r["provenance"].get("artifacts_resolved_at_add")),
            "metric_ranks": {mid: list(rk[rid]) for mid, rk in metric_ranks.items()
                             if rid in rk},
            "category_subscores": {c: list(s) for c, s in subscores.items()},
            "composite": list(composite) if composite else None,
        })
    overall = interval_rank_spreads(intervals) if intervals else {}
    for row in out_rows:
        row["rank"] = list(overall[row["id"]]) if row["id"] in overall else None
    out_rows.sort(key=lambda r: (r["composite"] is None,
                                 r["composite"] or [math.inf, math.inf],
                                 r["method"], r["version"]))

    # count-arm sub-board (§5.2): ranks only the methods that emit counts.
    count_m = next(m for m in registry["metrics"]
                   if m["arm"] == "count" and m["key"] == "crps")
    count_rows = [r for r in out_rows
                  if all(k in r["metrics"].get("count", {})
                         for k in ["crps", *count_m.get("companions", [])])]
    count_vals = {r["id"]: r["metrics"]["count"]["crps"] for r in count_rows}
    count_ranks = rank_spreads(count_vals, count_m["direction"], count_m["floor"]) \
        if count_vals else {}
    sub_count = {
        "note": cats["count_arm"]["note"],
        "rows": sorted(({"id": r["id"], "method": r["method"], "version": r["version"],
                         "metrics": {"count": r["metrics"]["count"]},
                         "rank": list(count_ranks[r["id"]])} for r in count_rows),
                       key=lambda r: (r["rank"], r["method"], r["version"])),
    }

    # CANDI-lineage sub-board (§5.3): covariate diagnostics, CANDI versions only.
    lineage_rows = [{"id": r["id"], "method": r["method"], "version": r["version"],
                     "date": r["date"], "metrics": {"diagnostics": r["metrics"]["diagnostics"]}}
                    for r in out_rows
                    if r["lineage"] == "candi" and r["metrics"].get("diagnostics")]
    sub_lineage = {"note": cats["covariate_diagnostics"]["note"],
                   "rows": sorted(lineage_rows, key=lambda r: (r["date"], r["id"]))}

    return {
        "categories_in_composite": composite_cats,
        "rows": out_rows,
        "unranked": sorted(({"id": f"{r['method']}@{r['version']}", "method": r["method"],
                             "version": r["version"],
                             "note": f"no {view}-view scores stamped"} for r in unranked),
                           key=lambda r: r["id"]),
        "sub_boards": {"count_arm": sub_count, "candi_lineage": sub_lineage},
    }


def compile_leaderboard(root: Path) -> Dict[str, Any]:
    registry = load_registry(root)
    boards = load_boards(root)
    out_boards: Dict[str, Any] = {}
    for bid, board in sorted(boards["boards"].items()):
        row_dir = root / "rows" / bid
        rows = []
        for path in sorted(row_dir.glob("*.json")) if row_dir.exists() else []:
            row = load_json(path)
            gate_row_shape(row, registry)
            gate_row_against_board(row, board, bid)
            if row["board"] != bid:
                raise GateError(f"{path} says board `{row['board']}` but lives under `{bid}`")
            rows.append(row)
        views = {view: compile_view(rows, registry, view)
                 for view in board.get("views", ["default"])}
        climb: Dict[str, List[Dict[str, Any]]] = {}
        for r in views["default"]["rows"]:
            climb.setdefault(r["method"], []).append(
                {"date": r["date"], "version": r["version"],
                 "composite": r["composite"], "lineage": r["lineage"]})
        for entries in climb.values():
            entries.sort(key=lambda e: (e["date"], e["version"]))
        out_boards[bid] = {"meta": board, "views": views, "climb": climb}
    return {
        "schema_version": 1,
        "registry": registry,
        "boards": out_boards,
        "reproducibility": {
            "score_command": "python -m candi.bench.external --store <regime.json> "
                             "--pred <pred_root> --out <scores.json>",
            "stamp_command": "python tools/leaderboard.py add <scores.json> --board <b> "
                             "--method <m> --version <v> ...",
            "note": "Every row links its score json and FIR path; scoring stays a manual Fir "
                    "run and this build only compiles what was stamped (PRD §6, §9).",
        },
    }


SITE_FILES = ("index.html", "app.js", "style.css")


def cmd_build(args: argparse.Namespace) -> int:
    root = Path(args.root)
    out = Path(args.out)
    compiled = compile_leaderboard(root)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "leaderboard.json").write_text(dump_json(compiled), encoding="utf-8")
    for name in SITE_FILES:
        src = root / "site" / name
        if not src.exists():
            raise GateError(f"site source {src} is missing")
        (out / name).write_bytes(src.read_bytes())
    print(f"built {out} ({len(compiled['boards'])} boards)")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    root = Path(args.root)
    registry = load_registry(root)
    boards = load_boards(root)
    n_rows = 0
    for bid, board in sorted(boards["boards"].items()):
        row_dir = root / "rows" / bid
        for path in sorted(row_dir.glob("*.json")) if row_dir.exists() else []:
            row = load_json(path)
            gate_row_shape(row, registry)
            gate_row_against_board(row, board, bid)
            # integrity: when the stamped score json is reachable here, its macro must still
            # say what the row says. Unreachable (CI has no cruxvault/results) is a skip.
            score_path = Path(row["provenance"]["score_json"])
            if not score_path.is_absolute():
                score_path = REPO / score_path
            if score_path.exists():
                pm = row["provenance"]["flags"].get("placement_method")
                source = placement_score(load_json(score_path, allow_nonfinite=True),
                                         pm, score_path) if pm else load_json(score_path)
                fresh, _ = extract_metrics(source, registry, allow_missing=True)
                stamped = {slot: block for slot, block in row["metrics"].items() if block}
                if fresh != stamped:
                    raise GateError(f"{path} no longer matches its score json {score_path}")
            n_rows += 1
    for name in SITE_FILES:
        src = root / "site" / name
        if not src.exists():
            raise GateError(f"site source {src} is missing")
        # the W3C namespace identifier (SVG createElementNS / favicon xmlns) is a name,
        # not a request; everything else that smells like a URL is refused.
        text = src.read_text(encoding="utf-8") \
            .replace("http://www.w3.org/", "").replace("http%3A%2F%2Fwww.w3.org%2F", "")
        if "http://" in text or "https://" in text:
            raise GateError(f"{src} references an external URL — the site makes no "
                            "external requests (PRD §8)")
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        for target in (a, b):
            ns = argparse.Namespace(root=root, out=target)
            cmd_build(ns)
        for name in ("leaderboard.json", *SITE_FILES):
            if (a / name).read_bytes() != (b / name).read_bytes():
                raise GateError(f"build is not deterministic: {name} differs between reruns")
    print(f"check passed: {n_rows} rows, {len(boards['boards'])} boards, "
          "deterministic rebuild bit-exact")
    return 0


# ---------------------------------------------------------------- CLI ---

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="leaderboard.py", description=__doc__.splitlines()[0])
    p.add_argument("--root", default=str(DEFAULT_ROOT),
                   help="leaderboard directory (registry, boards, rows, site)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="stamp one score json into one row")
    a.add_argument("score", help="candi.bench(.external)-shaped score json")
    a.add_argument("--board", required=True)
    a.add_argument("--method", required=True)
    a.add_argument("--version", required=True)
    a.add_argument("--date", required=True, help="YYYY-MM-DD the scores landed")
    a.add_argument("--lineage", required=True, choices=LINEAGES)
    a.add_argument("--position-class", required=True, choices=sorted(POSITION_CLASSES))
    a.add_argument("--cell-class", required=True, choices=sorted(CELL_CLASSES))
    a.add_argument("--scoring-sha", required=True, help="git SHA of the scoring code")
    a.add_argument("--store-manifest-hash", required=True,
                   help="hash of the store manifest the run scored against")
    a.add_argument("--fir-path", default=None,
                   help="run directory on Fir; default reads FIR_PATH.txt beside the score json")
    a.add_argument("--strict-score", default=None,
                   help="score json for the strict view (main board: P2 minus chr19)")
    a.add_argument("--placement-method", default=None,
                   help="the score file is a t54 Dataset-3 placement.json; stamp this "
                        "method's macro_all (the file must digest to the board's frozen "
                        "regime hash)")
    a.add_argument("--allow-missing", action="store_true",
                   help="record absent registry metrics instead of refusing")
    a.add_argument("--force", action="store_true", help="restamp an existing row")
    a.set_defaults(func=cmd_add)

    b = sub.add_parser("build", help="compile rows into _site/")
    b.add_argument("--out", default=str(DEFAULT_OUT))
    b.set_defaults(func=cmd_build)

    c = sub.add_parser("check", help="re-run every gate and diff a double build bit-exact")
    c.set_defaults(func=cmd_check)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
