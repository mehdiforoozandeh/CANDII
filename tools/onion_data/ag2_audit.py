"""Leakage audit: the AlphaGenome 2 (AG2) track status of every EIC track (onion plan, Amendment 5, B.1).

Unit: one non-control row of the EIC accession ledger. Control rows (`assay_class == control`) are skipped.

Rule (ported from Max's S12 `scripts/split/make_split_f1.py` l.11-17 and its `conditions_of`):
1. The track's `biosample_term_id` comes from the ENCODE catalog `experiments.tsv`, joined on `experiment`.
   A non-control track whose experiment has no term in the catalog is an error.
2. Condition: the condition whose `biosample_curies` is exactly the term ("equal"); else the composite condition
   (more than one CURIE) that contains the term and has the most ENCODE bulk tracks (`n_observed_encode_bulk`),
   ties broken by condition name ("composite"); else none.
3. Track status, first match wins:
   - `no_condition`: no condition contains the term (not queryable; no scRNA-seq conditioning, PI ruling 36).
   - `no_channel`: AG2 has no output channel for the mark (from `ag2_channels.tsv`).
   - `observed_matched`: AG2 has an observed training track for the condition, the mark and the target's read type.
   - `observed_other_read_type`: observed for the other read type only.
   - `sibling_condition`: not observed in the mapped condition, but observed in another condition that contains
     the same term.
   - `unseen`: the condition exists and has no observed track of the mark with either read type.
   "Observed" counts every row of `ag2_observed.tsv`, whatever its `data_source` (a training track from any source
   is a leak). An accessibility channel with an empty `endedness` matches both read types.
4. `api_label` is the expected label of the channel that the fetch scores as "matched" (the channel of the
   target's read type, else the other one; plan section A): `observed` when that channel is observed in the
   condition; else `imputed_supported` when the condition has an observed accessibility track, else
   `imputed_exploratory`. Empty for `no_condition` and `no_channel`.
5. `queryable` is 1 for the first four statuses of B.1 (`observed_matched`, `observed_other_read_type`,
   `sibling_condition`, `unseen`).

Comparison sets (`ag2_fair_sets.tsv`; PI ruling 34): `unseen_by_both` = status `unseen` and not trained on by the
model; `ag2_trained` = status `observed_matched` or `observed_other_read_type` and not trained on by the model.
Model `final` trained on the train role (sets on val and blind); `final_tv` and the published EIC entries
(`published_eic`) trained on train and val (sets on blind only). A set meets the contrast minimum at 10 tracks
from at least 3 cells.

`fetch_conditions.tsv`: per condition with a queryable validation or blind track, the count of such tracks per
role, and `needs_accessibility` = 1 when one of them is a DNase-seq or ATAC-seq track with status `unseen`
(decision 10: an accessibility track is fetched only when unseen).

Usage:
  ag2_audit.py run --ledger <tsv> --experiments <experiments.tsv> --observed <ag2_observed.tsv>
                   --conditions <ag2_conditions.tsv> --channels <ag2_channels.tsv> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ledger assay -> AG2 output type and canonical mark. Every panel histone of encode_catalog.PANEL_HISTONES is
# listed; a ledger assay outside this table is an error, never a silent `no_channel`.
ACCESSIBILITY = {"DNase-seq": "DNASE", "ATAC-seq": "ATAC"}
PANEL_HISTONES = (
    "H2AFZ H2AK5ac H2AK9ac H2BK120ac H2BK12ac H2BK15ac H2BK20ac H2BK5ac H3F3A H3K14ac H3K18ac "
    "H3K23ac H3K23me2 H3K27ac H3K27me3 H3K36me3 H3K4ac H3K4me1 H3K4me2 H3K4me3 H3K56ac H3K79me1 "
    "H3K79me2 H3K9ac H3K9me1 H3K9me2 H3K9me3 H3T11ph H4K12ac H4K20me1 H4K5ac H4K8ac H4K91ac"
).split()
# Names of one mark, on either side (ledger `assay`, AG2 `target`) -> one canonical name.
MARK_ALIASES = {"H2AFZ": "H2A.Z", "H2A.Z": "H2A.Z", "H2AZ": "H2A.Z", "H2A.Z.1": "H2A.Z",
                "H3F3A": "H3.3", "H3.3": "H3.3"}
MARK_MAP = {m: MARK_ALIASES.get(m, m) for m in PANEL_HISTONES}

STATUSES = ["observed_matched", "observed_other_read_type", "sibling_condition", "unseen", "no_channel",
            "no_condition"]
QUERYABLE = set(STATUSES[:4])
AG2_TRAINED = {"observed_matched", "observed_other_read_type"}
ENDS = {"single-ended": "single", "paired-ended": "paired"}
ASSAY_CLASSES = ["narrow_histone", "broad_histone", "accessibility_dnase", "accessibility_atac"]
# model -> (roles it trained on, splits it is scored on)
MODELS = [("final", {"train"}, ["val", "blind"]),
          ("final_tv", {"train", "val"}, ["blind"]),
          ("published_eic", {"train", "val"}, ["blind"])]
MIN_TRACKS, MIN_CELLS = 10, 3

AUDIT_COLS = ["eic_track", "role", "cell", "assay", "assay_class", "experiment", "run_type", "biosample_term_id",
              "condition", "condition_match", "status", "api_label", "matched_channel", "otherend_channel",
              "n_observed_channels_in_cell", "queryable"]
FAIR_COLS = (["split", "model", "track_set", "n_tracks", "n_cells", "meets_minimum"]
             + [f"n_{c}" for c in ASSAY_CLASSES] + ["per_assay"])
FETCH_COLS = ["condition", "n_val_tracks", "n_blind_tracks", "needs_accessibility"]


class AuditError(ValueError):
    pass


def read_tsv(path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path, cols, rows) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def channel_key(assay: str) -> tuple[str, str]:
    """(output_type, canonical mark) of a ledger assay; raises on an unmapped mark."""
    if assay in ACCESSIBILITY:
        return ACCESSIBILITY[assay], ""
    if assay in MARK_MAP:
        return "CHIP_HISTONE", MARK_MAP[assay]
    raise AuditError(f"unmapped mark {assay!r}: add it to MARK_MAP in ag2_audit.py")


def _ag2_key(output_type: str, target: str) -> tuple[str, str]:
    ot = output_type.replace("OutputType.", "").strip().upper()
    return ot, ("" if ot != "CHIP_HISTONE" else MARK_ALIASES.get(target.strip(), target.strip()))


class Ag2Meta:
    """The three tables of pinned interface G, indexed for the audit."""

    def __init__(self, observed: list[dict], conditions: list[dict], channels: list[dict]):
        self.curies = {c["condition"]: [x for x in c["biosample_curies"].split(";") if x] for c in conditions}
        self.n_bulk = {c["condition"]: int(c["n_observed_encode_bulk"] or 0) for c in conditions}
        self.has_acc = {c["condition"]: c["has_observed_accessibility"].strip() == "1" for c in conditions}
        # (output_type, mark) -> {endedness: channel}, lowest channel_index first
        self.channels: dict[tuple, dict[str, str]] = defaultdict(dict)
        for ch in sorted(channels, key=lambda r: int(r["channel_index"])):
            self.channels[_ag2_key(ch["output_type"], ch["target"])].setdefault(ch["endedness"].strip(), ch["channel"])
        # (condition, output_type, mark) -> set of observed endedness ("" = any); condition -> observed channels
        self.seen: dict[tuple, set] = defaultdict(set)
        self.obs_channels: dict[str, set] = defaultdict(set)
        for o in observed:
            ot, mark = _ag2_key(o["output_type"], o["target"])
            self.seen[(o["condition"], ot, mark)].add(o["endedness"].strip())
            self.obs_channels[o["condition"]].add(o["channel"])
        unknown = set(self.obs_channels) - set(self.curies)
        if unknown:
            raise AuditError(f"observed conditions missing from the conditions table: {sorted(unknown)[:5]}")

    def condition_of(self, term: str) -> tuple[str, str]:
        """(condition, condition_match) for an ENCODE ontology term."""
        cand = [c for c, cs in self.curies.items() if term and term in cs]
        equal = [c for c in cand if self.curies[c] == [term]]
        if equal:
            return max(sorted(equal), key=lambda c: self.n_bulk[c]), "equal"
        comp = sorted(c for c in cand if len(self.curies[c]) > 1)
        if comp:
            return max(comp, key=lambda c: self.n_bulk[c]), "composite"
        return "", "none"

    def ends_seen(self, cond: str, key: tuple) -> set:
        return self.seen.get((cond, *key), set())


def _covers(ends: set, end: str) -> bool:
    return end in ends or "" in ends


def audit_track(row: dict, term: str, meta: Ag2Meta) -> dict:
    key = channel_key(row["assay"])
    end = ENDS.get(row["run_type"])
    if end is None:
        raise AuditError(f"{row['eic_track']}: unknown run_type {row['run_type']!r}")
    other = "paired" if end == "single" else "single"
    cond, how = meta.condition_of(term)
    chans = meta.channels.get(key, {})
    matched = chans.get(end) or chans.get("", "")
    otherend = chans.get(other, "")
    out = {k: row.get(k, "") for k in ("eic_track", "role", "cell", "assay", "assay_class", "experiment",
                                         "run_type")}
    out.update(biosample_term_id=term, condition=cond, condition_match=how, matched_channel=matched,
               otherend_channel=otherend, api_label="",
               n_observed_channels_in_cell=len(meta.obs_channels.get(cond, ())) if cond else 0)
    ends = meta.ends_seen(cond, key) if cond else set()
    if not cond:
        status = "no_condition"
    elif not chans:
        status = "no_channel"
    elif _covers(ends, end):
        status = "observed_matched"
    elif ends:
        status = "observed_other_read_type"
    elif any(meta.ends_seen(c, key) for c, cs in meta.curies.items() if c != cond and term in cs):
        status = "sibling_condition"
    else:
        status = "unseen"
    if status in QUERYABLE:
        scored_end = end if matched else other            # the channel scored as "matched" (plan section A)
        if _covers(ends, scored_end):
            out["api_label"] = "observed"
        else:
            out["api_label"] = "imputed_supported" if meta.has_acc.get(cond) else "imputed_exploratory"
    out["status"] = status
    out["queryable"] = int(status in QUERYABLE)
    return out


def audit(ledger: list[dict], experiments: list[dict], meta: Ag2Meta) -> list[dict]:
    terms = {e["experiment"]: e.get("biosample_term_id", "").strip() for e in experiments}
    rows = [r for r in ledger if r["assay_class"] != "control"]
    for r in rows:                                          # fail on every unmapped mark before any output
        channel_key(r["assay"])
    missing = sorted({r["experiment"] for r in rows if not terms.get(r["experiment"])})
    if missing:
        raise AuditError(f"{len(missing)} experiments without a biosample_term_id in the catalog: {missing[:5]}")
    return [audit_track(r, terms[r["experiment"]], meta) for r in rows]


def _set_row(split, model, name, tracks) -> dict:
    cells = {t["cell"] for t in tracks}
    by_cls = Counter(t["assay_class"] for t in tracks)
    by_assay = Counter(t["assay"] for t in tracks)
    return {"split": split, "model": model, "track_set": name, "n_tracks": len(tracks), "n_cells": len(cells),
            "meets_minimum": int(len(tracks) >= MIN_TRACKS and len(cells) >= MIN_CELLS),
            **{f"n_{c}": by_cls.get(c, 0) for c in ASSAY_CLASSES},
            "per_assay": ";".join(f"{a}:{n}" for a, n in sorted(by_assay.items()))}


def fair_sets(rows: list[dict]) -> list[dict]:
    out = []
    for model, trained, splits in MODELS:
        for split in splits:
            pool = [r for r in rows if r["role"] == split and r["role"] not in trained]
            out.append(_set_row(split, model, "unseen_by_both", [r for r in pool if r["status"] == "unseen"]))
            out.append(_set_row(split, model, "ag2_trained", [r for r in pool if r["status"] in AG2_TRAINED]))
    return out


def fetch_conditions(rows: list[dict]) -> list[dict]:
    agg: dict[str, dict] = {}
    for r in rows:
        if r["role"] not in ("val", "blind") or not r["queryable"]:
            continue
        a = agg.setdefault(r["condition"], {"condition": r["condition"], "n_val_tracks": 0, "n_blind_tracks": 0,
                                            "needs_accessibility": 0})
        a[f"n_{r['role']}_tracks"] += 1
        if r["assay"] in ACCESSIBILITY and r["status"] == "unseen":
            a["needs_accessibility"] = 1
    return [agg[c] for c in sorted(agg)]


def summary_lines(rows: list[dict]) -> list[str]:
    cnt = Counter((r["role"], r["status"]) for r in rows)
    return [f"role={role} status={s} n={cnt[(role, s)]}"
            for role in ("train", "val", "blind") for s in STATUSES if cnt[(role, s)]]


def run(args) -> int:
    meta = Ag2Meta(read_tsv(args.observed), read_tsv(args.conditions), read_tsv(args.channels))
    rows = audit(read_tsv(args.ledger), read_tsv(args.experiments), meta)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_tsv(out / "ag2_eic_audit.tsv", AUDIT_COLS, rows)
    write_tsv(out / "ag2_fair_sets.tsv", FAIR_COLS, fair_sets(rows))
    write_tsv(out / "fetch_conditions.tsv", FETCH_COLS, fetch_conditions(rows))
    for line in summary_lines(rows):
        print(line)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="audit every non-control ledger track")
    for name in ("ledger", "experiments", "observed", "conditions", "channels", "out"):
        p.add_argument(f"--{name}", required=True)
    args = ap.parse_args(argv)
    try:
        return run(args)
    except AuditError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
