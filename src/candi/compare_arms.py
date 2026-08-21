"""Paired arm-vs-arm comparison for the cell-identity conditioning experiment.

    python -m candi.compare_arms --case run_cell-id.json --control run_cell-off.json
    python -m candi.compare_arms --case A.json --control B.json --control C.json --md out.md
    python -m candi.compare_arms --case A.json --control B.json \
        --ablation-row assay_id --ablation-row run_type

WHAT THIS COMPARES, AND WHY IT IS PAIRED
Evaluation is deterministic — `shuffle=False`, `dsf_sampling="off"`, and eval unit i always draws
the same window batch — so every arm scores the SAME positions on the SAME targets. The delta per
target is therefore a paired quantity, which is where the power comes from: comparing two
independent means over 96 targets would throw most of it away.

THE UNIT IS THE TARGET, NOT THE POSITION
`(T_biosample, imp_biosample, assay)` — 96 of them on the full EIC panel. `eval.py` records the
reason in `_cluster_bootstrap_ci`: position-level intervals ran ~24x too narrow, because positions
inside one target are not independent draws. That helper is reused here verbatim rather than
reimplemented; it is `n_fg`-weighted, drops exact ties from the sign test, and reports direction
separately from significance (a large effect in the WRONG direction still "excludes zero").

SIGN CONVENTION
CRPS is a loss, so `delta = control - case`. POSITIVE means the case arm is BETTER. That makes
`supports_direction` (lo > 0) read directly as "the case arm won".

SIGN CONVENTION, INVERTED, FOR THE ABLATION TABLES (`--ablation-row`)
`d_eta` is NOT a loss. It is how far the decoder's output MOVES when one covariate row is swapped for
another target's real value, so LARGER IS MORE STEERING and larger is what the case arm wants. The
ablation path therefore uses `delta = case - control` -- the OPPOSITE subtraction from the CRPS path
above -- precisely so that POSITIVE still reads as "the case arm won" and `supports_direction` keeps
its one meaning across the whole report. A reader who carries the CRPS subtraction over to these
tables will read every ablation sign backwards, so every ablation line is labelled with its own
`delta = case - control` in the printed output as well as here.

WHAT THE ABLATION PATH COMPARES
`M2.ablation.<row>` for row in {depth, assay_id, read_length, run_type}: the sentinel-free
cross-target metadata swap. Its `per_target` list holds one record per EVAL UNIT (~1,123 on the full
panel), not one per target -- ~11.7 units share each of the 96 `(biosample, imp_biosample, assay)`
targets. Each arm is therefore collapsed to ONE `n_fg`-weighted value per target key first, exactly
the way `_cluster_bootstrap_ci` collapses internally, and only then are the two arms paired on that
key. Pairing the raw lists positionally would silently align records from different targets.

ABLATION CAVEATS THAT ARE REPORTED, NOT ASSUMED
  * `mode` must be `cross_target`. The `within_batch` variant is a STRUCTURAL NULL -- `meta_dsf{k}`
    has no window axis, so a within-batch permutation is the identity and reads 0 for every model.
    Comparing two arms on it compares two zeros. It is warned about loudly and flagged `mode_ok`.
  * `n_sentinel_skipped` -- targets whose covariate row held MISSING/CLOZE and were never probed.
  * `purity_fallback_fired` -- the foreground fell off the `target >= 1` purity filter, so the swap
    was scored on background. It fires on ~18% of real held-out records, and it is invisible unless
    counted, so the record counts for both arms are printed. They are SURFACED, NOT DROPPED: the
    CRPS path passes no flag to `_cluster_bootstrap_ci` either, and dropping at the collapsed-target
    level would discard a whole target for one bad unit. Pass `exclude_purity_fallback=True` to drop
    the flagged records before collapsing instead.

WHAT IS REPORTED, per PI ruling
  * `crps`               PRIMARY. The headline number.
  * `crps_oracle_scaled` The same score after refitting the optimal per-assay multiplicative scale,
                         so it is blind to a pure scale win.
  * `scale_error`        `crps - crps_oracle_scaled`: how much of the arm's loss is fixable scale.
A gain present in `crps` but absent in `crps_oracle_scaled` means the embedding learned HOW MUCH
signal a cell type carries, not WHERE it sits. Both are reported; neither is collapsed into the
other.

STRATIFICATION
V_ and B_ targets are reported separately and never pooled — on the CRPS tables and on the ablation
tables alike. The B_ cells are the challenge's blind test set, drawn deliberately from poorly
characterised biosamples — 9 of the 12 carry <=2 T_ tracks — so they are simultaneously where the
embedding has the least to learn from and where imputation matters most. An average over both would
hide whichever effect is smaller.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# t22: the statistic lives in `candi.stats` now — `candi.eval` is being deleted (D15)
from candi.stats import cluster_bootstrap_ci as _cluster_bootstrap_ci

METRICS = ("crps", "crps_oracle_scaled", "scale_error")
LOWER_IS_BETTER = {"crps": True, "crps_oracle_scaled": True, "scale_error": True}

# `M2.ablation` rows, in `eval.META_ROWS` order.
ABLATION_ROWS = ("depth", "assay_id", "read_length", "run_type")
# Steering magnitudes: mean and max |d eta| over the foreground. LARGER IS BETTER on both.
ABLATION_METRICS = ("d_eta", "d_eta_max")
# The only mode whose numbers mean anything across arms; see the docstring.
ABLATION_MODE = "cross_target"


def _load(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    if "M1" not in d:
        raise ValueError(f"{path} has no M1 block — is it a candi run json?")
    return d


def _arm_label(d: dict, path: str) -> str:
    cfg = d.get("config", {})
    return cfg.get("tag") or f"{cfg.get('cell_cond', '?')}@{Path(path).stem}"


def _targets(d: dict, block: str) -> Dict[str, dict]:
    pt = d["M1"].get(f"{block}_per_target")
    if pt is None:
        raise ValueError(
            f"run json has no M1.{block}_per_target. It was produced by an eval build predating "
            "per-target reporting; re-run evaluation before comparing arms.")
    return pt


def _stratum(key: str) -> str:
    """`T_x|V_x|assay` -> 'V' or 'B' from the imputation target's prefix."""
    parts = key.split("|")
    return parts[1][:1] if len(parts) >= 2 and parts[1][:1] in ("V", "B") else "?"


def compare(case: dict, control: dict, *, block: str = "imp", metric: str = "crps",
            n_boot: int = 1000, seed: int = 0,
            stratum: Optional[str] = None) -> dict:
    """Paired per-target delta (control - case) with a target-clustered bootstrap."""
    ct, kt = _targets(case, block), _targets(control, block)
    shared = sorted(set(ct) & set(kt))
    if stratum:
        shared = [k for k in shared if _stratum(k) == stratum]
    records: List[Dict] = []
    for k in shared:
        a, b = ct[k], kt[k]
        va, vb = a.get(metric), b.get(metric)
        if va is None or vb is None:
            continue
        try:
            if not (float(va) == float(va) and float(vb) == float(vb)):   # NaN guard
                continue
        except (TypeError, ValueError):
            continue
        # weight by the SMALLER of the two point counts: a target the two arms scored on different
        # numbers of positions is not comparable at face value, and the smaller n is the honest one.
        w = min(int(a.get("n_points", 0) or 0), int(b.get("n_points", 0) or 0))
        records.append(dict(target=tuple(k.split("|")), mean_delta=float(vb) - float(va), n_fg=max(w, 1)))
    out = _cluster_bootstrap_ci(records, value_key="mean_delta", n_boot=n_boot, seed=seed)
    out["metric"] = metric
    out["block"] = block
    out["stratum"] = stratum or "all"
    out["n_targets_shared"] = len(shared)
    out["n_targets_dropped_nan"] = len(shared) - len(records)
    return out


# ---------------------------------------------------------------------------
# metadata-ablation steering (M2.ablation.<row>) — see the SIGN CONVENTION note in the module
# docstring: this path subtracts the OTHER WAY ROUND from the CRPS path above.
# ---------------------------------------------------------------------------

def _ablation_block(d: dict, row: str, path: str = "?") -> dict:
    """`M2.ablation.<row>`, with a message that says what to do when it is not there."""
    if row not in ABLATION_ROWS:
        raise ValueError(f"unknown ablation row {row!r}; expected one of {list(ABLATION_ROWS)}")
    abl = (d.get("M2") or {}).get("ablation")
    if not isinstance(abl, dict):
        raise ValueError(
            f"{path} has no M2.ablation block. It was produced by an eval build predating the "
            "sentinel-free metadata ablation; re-run evaluation before comparing steering.")
    b = abl.get(row)
    if not isinstance(b, dict):
        raise ValueError(f"{path} has no M2.ablation.{row}; rows present: {sorted(abl)}")
    return b


def _ablation_key(rec: dict) -> Optional[str]:
    """`(biosample, imp_biosample, assay)` -> `'a|b|c'`, matching M1's per-target key format.

    `train._jsonable` turns a tuple DICT KEY into `'a|b|c'` but a tuple VALUE into a JSON LIST, and
    `target` is a value here — so real files carry the list form. Both are accepted so the reader
    does not have to know which one it is holding.
    """
    t = rec.get("target")
    if isinstance(t, str):
        return t if t.count("|") == 2 else None
    if isinstance(t, (list, tuple)) and len(t) == 3:
        return "|".join(str(x) for x in t)
    return None


def _ablation_by_target(block: dict, metric: str, *,
                        exclude_purity_fallback: bool = False) -> Tuple[Dict[str, Tuple[float, float]], Dict]:
    """Collapse an arm's `per_target` records to ONE `n_fg`-weighted value per target key.

    The list holds one record per EVAL UNIT (~11.7 per target on the full panel), so this collapse is
    what makes the two arms pairable at all. It mirrors `_cluster_bootstrap_ci`'s own aggregation, so
    a collapsed value here equals the cluster value the bootstrap would have formed.
    """
    agg: Dict[str, List[float]] = {}
    stats = dict(n_records=0, n_bad_key=0, n_nan=0, n_purity_fallback_fired=0,
                 n_purity_fallback_excluded=0)
    for rec in block.get("per_target") or []:
        stats["n_records"] += 1
        k = _ablation_key(rec)
        if k is None:
            stats["n_bad_key"] += 1
            continue
        if bool(rec.get("purity_fallback_fired", False)):
            stats["n_purity_fallback_fired"] += 1
            if exclude_purity_fallback:
                stats["n_purity_fallback_excluded"] += 1
                continue
        try:
            v = float(rec.get(metric))
        except (TypeError, ValueError):
            stats["n_nan"] += 1
            continue
        if v != v:                                              # NaN guard
            stats["n_nan"] += 1
            continue
        w = max(float(rec.get("n_fg", 1) or 1), 1.0)
        s = agg.setdefault(k, [0.0, 0.0])
        s[0] += w * v
        s[1] += w
    return {k: (a / b, b) for k, (a, b) in agg.items() if b > 0}, stats


def compare_ablation(case: dict, control: dict, row: str, *, metric: str = "d_eta",
                     n_boot: int = 1000, seed: int = 0, stratum: Optional[str] = None,
                     exclude_purity_fallback: bool = False, warn_mode: bool = True,
                     case_path: str = "case", control_path: str = "control") -> dict:
    """Paired per-target steering delta (CASE - CONTROL) with a target-clustered bootstrap.

    NOTE THE SUBTRACTION. `d_eta` measures how far the output moves when the covariate is swapped, so
    larger is more steering and POSITIVE means the CASE arm steers more — i.e. positive still reads
    "the case arm won", but it is `case - control` here where the CRPS path is `control - case`.
    """
    cb = _ablation_block(case, row, case_path)
    kb = _ablation_block(control, row, control_path)
    modes = (str(cb.get("mode")), str(kb.get("mode")))
    mode_ok = all(m == ABLATION_MODE for m in modes)
    if not mode_ok and warn_mode:
        warnings.warn(
            f"M2.ablation.{row} mode is {modes[0]!r}/{modes[1]!r}, not {ABLATION_MODE!r}. "
            "within_batch is a STRUCTURAL NULL — it is the identity swap and reads 0 for every "
            "model — so an arm-vs-arm comparison on it compares two zeros and means nothing.",
            UserWarning, stacklevel=2)

    cv, cst = _ablation_by_target(cb, metric, exclude_purity_fallback=exclude_purity_fallback)
    kv, kst = _ablation_by_target(kb, metric, exclude_purity_fallback=exclude_purity_fallback)

    def _in(k: str) -> bool:
        return stratum is None or _stratum(k) == stratum

    ck = {k for k in cv if _in(k)}
    kk = {k for k in kv if _in(k)}
    shared = sorted(ck & kk)
    records: List[Dict] = []
    for k in shared:
        va, wa = cv[k]
        vb, wb = kv[k]
        # weight by the SMALLER of the two foreground totals, as the CRPS path does with n_points: a
        # target the two arms scored on different amounts of foreground is not comparable at face
        # value, and the smaller one is the honest weight.
        records.append(dict(target=tuple(k.split("|")), mean_delta=va - vb,
                            n_fg=max(int(min(wa, wb)), 1)))
    out = _cluster_bootstrap_ci(records, value_key="mean_delta", n_boot=n_boot, seed=seed)
    out.update(
        kind="ablation", row=row, metric=metric, stratum=stratum or "all",
        sign_convention="case - control (POSITIVE = case better = MORE steering; "
                        "INVERTED vs the CRPS tables)",
        mode_case=modes[0], mode_control=modes[1], mode_ok=bool(mode_ok),
        n_targets_case=len(ck), n_targets_control=len(kk), n_targets_paired=len(shared),
        n_targets_dropped_case_only=len(ck - kk), n_targets_dropped_control_only=len(kk - ck),
        n_targets_dropped_unmatched=len(ck ^ kk),
        n_sentinel_skipped_case=cb.get("n_sentinel_skipped"),
        n_sentinel_skipped_control=kb.get("n_sentinel_skipped"),
        n_records_case=cst["n_records"], n_records_control=kst["n_records"],
        n_records_nan_case=cst["n_nan"], n_records_nan_control=kst["n_nan"],
        n_purity_fallback_fired_case=cst["n_purity_fallback_fired"],
        n_purity_fallback_fired_control=kst["n_purity_fallback_fired"],
        purity_fallback_excluded=bool(exclude_purity_fallback),
        rollup_case=_ablation_rollup(cb), rollup_control=_ablation_rollup(kb))
    return out


def _ablation_rollup(block: dict) -> dict:
    """The arm's own scalar summary, carried through so the paired delta can be read in context."""
    return {k: block.get(k) for k in ("mean_abs_d_eta", "max_abs_d_eta", "mean_d_crps", "n_targets",
                                      "frac_true_better", "mode", "n_sentinel_skipped",
                                      "uses_covariate")}


def _fmt_abl_header(r: dict) -> List[str]:
    """The per-block caveat lines: the inverted sign, the mode, and the two invisible counts."""
    lines = ["  sign: delta = case - control; POSITIVE = case better = MORE steering "
             "(INVERTED vs the CRPS tables above)",
             f"  mode={r['mode_case']}/{r['mode_control']}  "
             f"n_sentinel_skipped={r['n_sentinel_skipped_case']}/{r['n_sentinel_skipped_control']}  "
             f"purity_fallback_fired={r['n_purity_fallback_fired_case']}/"
             f"{r['n_purity_fallback_fired_control']} of "
             f"{r['n_records_case']}/{r['n_records_control']} records"
             + ("  (EXCLUDED)" if r["purity_fallback_excluded"] else "  (surfaced, not dropped)"),
             f"  targets: paired={r['n_targets_paired']}  "
             f"dropped_unmatched={r['n_targets_dropped_unmatched']} "
             f"(case-only={r['n_targets_dropped_case_only']}, "
             f"control-only={r['n_targets_dropped_control_only']})"]
    if not r["mode_ok"]:
        lines.insert(0, f"  !! WARNING: mode is not {ABLATION_MODE} — within_batch is a STRUCTURAL "
                        "NULL (identity swap, reads 0 for every model). These numbers compare two "
                        "zeros and MUST NOT be read as a steering result.")
    return lines


def _fmt(r: dict) -> str:
    if not r["n_clusters"]:
        return f"  {r['stratum']:>4s}  (no comparable targets)"
    verdict = ("case better" if r["supports_direction"]
               else "control better" if r["hi"] < 0
               else "no difference")
    return (f"  {r['stratum']:>4s}  n={r['n_clusters']:>3d}  "
            f"delta={r['mean']:+.5f} [{r['lo']:+.5f}, {r['hi']:+.5f}]  "
            f"{r['n_pos']}+/{r['n_neg']}-/{r['n_tied']}=  p={r['sign_test_p']:.4f}  {verdict}")


def run(case_path: str, control_paths: Sequence[str], *, blocks=("imp", "den"),
        n_boot: int = 1000, seed: int = 0, ablation_rows: Sequence[str] = (),
        exclude_purity_fallback: bool = False) -> dict:
    case = _load(case_path)
    case_label = _arm_label(case, case_path)
    results: Dict[str, dict] = {}
    ablation: Dict[str, dict] = {}
    lines: List[str] = []

    for cpath in control_paths:
        control = _load(cpath)
        clabel = _arm_label(control, cpath)
        lines.append(f"\n### {case_label}  vs  {clabel}")
        lines.append("delta = control - case; POSITIVE = case better (CRPS is a loss)")
        for block in blocks:
            lines.append(f"\n{block.upper()}")
            for metric in METRICS:
                lines.append(f" {metric}")
                for stratum in (None, "V", "B"):
                    r = compare(case, control, block=block, metric=metric,
                                n_boot=n_boot, seed=seed, stratum=stratum)
                    results[f"{clabel}|{block}|{metric}|{r['stratum']}"] = r
                    lines.append(_fmt(r))

        for row in ablation_rows:
            lines.append(f"\nM2 ABLATION `{row}` — metadata steering")
            head = None
            for metric in ABLATION_METRICS:
                for stratum in (None, "V", "B"):
                    r = compare_ablation(case, control, row, metric=metric, n_boot=n_boot,
                                         seed=seed, stratum=stratum,
                                         exclude_purity_fallback=exclude_purity_fallback,
                                         case_path=case_path, control_path=cpath,
                                         warn_mode=(head is None))
                    key = f"{clabel}|ablation|{row}|{metric}|{r['stratum']}"
                    results[key] = r
                    ablation[key] = r
                    if head is None:
                        head = r
                        lines.extend(_fmt_abl_header(r))
                    if stratum is None:
                        lines.append(f" {metric}  [delta = case - control; +ve = case better = "
                                     "MORE steering]")
                    lines.append(_fmt(r))
    return dict(case=case_label, results=results, ablation=ablation, text="\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True, help="the conditioned arm (e.g. --cell-cond id)")
    ap.add_argument("--control", required=True, action="append",
                    help="repeatable: the off arm, the random arm, ...")
    ap.add_argument("--block", default=["imp", "den"], action="append", dest="blocks_extra",
                    help=argparse.SUPPRESS)
    ap.add_argument("--imp-only", action="store_true", help="skip the denoising block")
    ap.add_argument("--ablation-row", action="append", default=[], dest="ablation_rows",
                    choices=list(ABLATION_ROWS),
                    help="repeatable: also compare M2.ablation.<row> steering (d_eta, d_eta_max). "
                         "NOTE the delta is case - control there, the OPPOSITE of the CRPS tables, "
                         "because more steering is better.")
    ap.add_argument("--ablation-exclude-purity-fallback", action="store_true",
                    help="drop per-target records with purity_fallback_fired before collapsing "
                         "(default: keep them and print the count)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--md", default=None, help="also write the table to a markdown file")
    a = ap.parse_args()

    blocks = ("imp",) if a.imp_only else ("imp", "den")
    out = run(a.case, a.control, blocks=blocks, n_boot=a.n_boot, seed=a.seed,
              ablation_rows=tuple(a.ablation_rows),
              exclude_purity_fallback=a.ablation_exclude_purity_fallback)
    print(out["text"])
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out["results"], indent=2))
    if a.md:
        Path(a.md).write_text(f"# Arm comparison — case `{out['case']}`\n\n```\n{out['text']}\n```\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
