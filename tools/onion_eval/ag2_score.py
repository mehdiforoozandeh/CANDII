"""The same-track scorer of the AlphaGenome 2 (AG2) comparison and the `ag2_board` table (chunk AGc;
onion plan, Amendment 5, sections B.1 to B.6).

    ag2_score.py score --evalset <json> --audit <ag2_eic_audit.tsv> --tables <dir>
        --truth-from <run or baseline method track dir> --method <name>=<dir>[,<dir>...] [...]
        --ag2 <ag2 pack dir> --peaks <dir of narrowPeak> --split val|blind [--allow-blind]
        [--repeat-blind-reason "<text>"] [--table headline|control_free] [--n-boot 2000]
        [--allow-unscored] --out <dir>
    -> <out>/per_target.tsv, ag2_board_<split>.tsv, ag2_board_excluded_<split>.tsv,
       ag2_contrasts_<split>.tsv, unscored_targets.tsv, SCORED

    ag2_score.py gate --targets <tsv> --pred <Max's ag2pred dir> --store <CANDI_STORE dir>
        --tiles <dir> --reference <Max's per-target tsv> [--tracks <Max's tracks tsv>]
        [--chroms chr20,chr21,chr22] --out <dir>
    -> <out>/gate.tsv, gate_skipped.tsv and the printed lines `targets compared: k of n`,
       `<metric>: max |diff| ...` and `gate: PASS|FAIL`

Inputs of `score` are method track files (pinned interface F). A method directory is `<dir>/<regime>/`
with `tracks.json`, or one regime directory. `--method name=d1,d2,d3` gives the seeds of one method:
each per-target metric is the mean over the seeds, and `rspec_cal_seed_range` is max minus min.
`--ag2` gives two rows, `ag2_matched` and `ag2_otherend`. The `training_mean_signal` method is
required: it is the `a` of `metrics.r_spec`.

Metrics per method and target, on all bins of the eval regions (decision 6), ported from Max's S12
`scripts/api/score_f1.py` (GitHub mlibbrecht/01_2026-10-19_research at bde905a,
projects/03_2026-10-20_epi_imputation/sprints/12_2026-11-12_ag2_fair_test):
- `rspec_cal`, `pearson_cal`: after `ag2_space.isotonic_recalibrate` of the prediction to the truth
  (l.281). `rspec`, `pearson`: on the raw prediction. r_spec by `metrics.r_spec` with
  `a` = the target's `training_mean_signal` track; that method's own r_spec columns are 0 by
  definition (score_f1.py l.282-283).
- `spearman`: on Max's fixed subsample of 2,000,000 bins drawn with seed 0 when the track is longer
  (l.100-101, l.280).
- `rspec_cal_128bp`: truth, recalibrated prediction and `a` averaged to the genome-aligned 128 bp grid
  (AG2's own bins) inside each region by `ag2_space.resample_mean`.
- Peaks by `ag2_peaks`: 500 bp tiles from each region's start; score = the tile mean of the raw
  prediction (l.273); consensus keyed by cell (Max merged replicate biosamples of one term, l.127-131).
  Two truth sources (`truth_peaks`): `encode` (the ENCODE peak files `<peaks>/<eic_track>.narrowPeak.gz`;
  tile set = the base tracks of the target's assay of roles train and val, plus blind on the blind
  split, second experiments removed) and `pipeline_derived` (bins of the truth with -log10 p >= 2; the
  tile set is the scored targets of the assay, because a training track has no `signal.h5`).
  `auprc` (Max's unweighted mean over tracks), `auprc_posfrac` (the weight of the paper's
  `auprc_weighted`), `auprc_f1` .. `auprc_f8` (8 peak-frequency bins).
- Binary call of decision 17 (`peak_*`, oracle threshold): a tile is called when any bin of the
  recalibrated prediction is >= asinh(2). `auprc_peakprob` and `nbpeak_*` (call at probability 0.5)
  from `peakprob` (tile max); `not applicable` for kinds that give no probability.
- `fe_pearson_25bp`, `fe_spearman_25bp`: the `native` array against `native_truth` of `--truth-from`
  (secondary space A or B; headline table only).

`--table headline` reads regime `leave_one_assay_out`, `<id>.signal.npy`; `--table control_free` reads
regime `accessibility_only` and, for the `run` and `baseline` kinds, `<id>.signal_pc.npy` (`not
available` when absent). The other kinds use no control and read `signal.npy` in both tables.

Track sets (section B.1, PI ruling 34): a target of status `unseen` is in `unseen_by_both` and one of
status `observed_matched` or `observed_other_read_type` is in `ag2_trained`, for every method that did
not train on it; AG2 rows take the set from the status alone. Other statuses enter no set and are
listed in `unscored_targets.tsv`. `trained_on_track`: kind `run` and `baseline` trained on the roles of
`tracks.json` key `trained_on_roles` (list or comma string; required for a run, default `train` for a
baseline); `external` on train and
val; `oracle_baseline` and `replicate_floor` on none; AG2 from the audit status. Contrasts are
`method - ag2_matched` per target, per set (never pooled), with a cluster bootstrap over cells
(`summarize_f1.py` `boot_ci`, l.46-53); a set under 10 tracks or 3 cells prints
`too few tracks for a contrast` in place of an interval. The board intervals use the same rule.
`pseudoreplicate_half` enters no contrast (its truth is the other half). Column `preregistered` is 1
only for `minicandi_final - ag2_matched`, `rspec_cal`, `unseen_by_both`, assay class `all`.

Board means (`ag2_board_<split>.tsv`): for each (track set, assay class, truth source, metric), every
method's mean is taken on the common tracks, i.e. the tracks where every method of the board that has
a number for that metric has a finite value; `n_<metric>` prints that count. Each excluded (track,
method, metric) case is listed in `ag2_board_excluded_<split>.tsv`.

`uses_target_cell_control` comes per track from `lambda_source` of `tracks.json` (a string or
{measurement_id: string}): `observed_control` and `pipeline_signal` give `yes`; `predicted_control`,
`training_average_control`, `local_background`, `hotspot2_density`, `none` and
`training_tracks_own_observed_control` give `no`; another value is refused. In the control_free table
a run or baseline reads `lambda_source_predicted_control` (default `predicted_control`). An external
entry must state `uses_target_cell_control` (yes|no) in its `tracks.json`, else it is refused.

Refusals: a method directory whose `regions_sha256` or `region_bins` differs from the eval set's, or
whose `split` differs, or that lacks a scored target (kind `replicate_floor` apart: a target without
halves prints `not available`); two truth arrays of one target that differ (`replicate_floor` apart);
a target listed in two track sets; an AG2 directory without `EVALUATION_ONLY`; a run directory
without `trained_on_roles`; an eval-set target of the split and regime that `--truth-from` lacks
(unless `--allow-unscored`, which lists it in `unscored_targets.tsv` with the reason). `evalset_sha256` is
recorded per method, never compared. `--split blind` needs `--allow-blind`; it appends `started` and
`scored` rows (`score.ledger_append`) to `BLIND_PASSES.tsv` beside the eval set under the key
`ag2:<resolved ag2 pack dir>`, and refuses a second pass without `--repeat-blind-reason`.

AG2 outputs are evaluation data only (decision 12): this tool reports them and writes nothing that a
training run, a baseline or a selection reads.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_eval import ag2_peaks as PK  # noqa: E402
from onion_eval import ag2_space as SP  # noqa: E402
from onion_eval import evalset as E  # noqa: E402
from onion_eval import metrics as Mx  # noqa: E402
from onion_eval.ag2_baselines import regions_sha256  # noqa: E402  (one form for every writer and reader)

BIN_BP = 25
AG2_BIN_BP = 128
NSUB = 2_000_000                    # score_f1.py l.42
NCAL = 2000                         # score_f1.py l.42
PEAK_CALL = float(np.arcsinh(2.0))  # decision 17: recalibrated value >= asinh(2), p <= 0.01
PROB_CALL = 0.5
MIN_TRACKS, MIN_CELLS = 10, 3
TOO_FEW = "too few tracks for a contrast"
NA_APPL = "not applicable"
NA_AVAIL = "not available"
LEDGER_PREFIX = "ag2:"
MEAN_METHOD = "training_mean_signal"
HALF_METHOD = "pseudoreplicate_half"
AG2_MATCHED, AG2_OTHER = "ag2_matched", "ag2_otherend"
TABLE_REGIME = {"headline": "leave_one_assay_out", "control_free": "accessibility_only"}
UNSEEN, AG2_TRAINED = "unseen_by_both", "ag2_trained"
SET_OF_STATUS = {"unseen": UNSEEN, "observed_matched": AG2_TRAINED,
                 "observed_other_read_type": AG2_TRAINED}
AG2_TRAINED_STATUS = {"observed_matched", "observed_other_read_type", "observed_by_name",
                      "sibling_condition"}
COMPARISON_LABEL = {UNSEEN: "headline: neither trained on these tracks",
                    AG2_TRAINED: "ag2_trained_on: AG2 trained on these tracks, the count model did not"}
NO_SET = "none"
TRUTH_PEAKS = ("encode", "pipeline_derived")
KINDS = ("run", "baseline", "external", "ag2", "oracle_baseline", "replicate_floor")
NB_KINDS = ("run", "baseline")

PEAK_METRICS = (["auprc", "auprc_posfrac"] + [f"auprc_f{k}" for k in range(1, 9)]
                + ["auprc_peakprob", "peak_precision", "peak_recall", "peak_f1",
                   "nbpeak_precision", "nbpeak_recall", "nbpeak_f1"])
SIGNAL_METRICS = ["rspec_cal", "pearson_cal", "spearman", "rspec", "pearson", "rspec_cal_128bp",
                  "fe_pearson_25bp", "fe_spearman_25bp"]
METRICS = SIGNAL_METRICS + PEAK_METRICS
CI_METRICS = ["rspec_cal", "pearson_cal", "spearman", "auprc"]
CONTRAST_METRICS = ["rspec_cal", "pearson_cal", "spearman", "auprc", "rspec_cal_128bp"]
#: lambda_source of a track -> whether its signal used the target cell's observed control
CONTROL_OF_LAMBDA = {"observed_control": "yes", "pipeline_signal": "yes",
                     "predicted_control": "no", "training_average_control": "no",
                     "local_background": "no", "hotspot2_density": "no", "none": "no",
                     "training_tracks_own_observed_control": "no"}
PREREGISTERED = ("minicandi_final", AG2_MATCHED, "rspec_cal", UNSEEN)
LABEL_COLS = ["trained_on_cell_other_tracks", "reads_target_cell_inputs_at_test_positions",
              "uses_target_cell_control", "truth_peaks", "recalibration", "noise_floor_row"]
PER_TARGET_COLS = (["method", "kind", "table", "regime", "target", "eic_track", "cell", "assay",
                    "assay_class", "role", "status", "api_label", "track_set", "comparison_label",
                    "trained_on_track", "lambda_source", "n_seeds"] + METRICS + ["rspec_cal_seed_range"]
                   + LABEL_COLS)
BOARD_METRICS = ["rspec_cal", "pearson_cal", "spearman", "auprc", "auprc_weighted"] + \
    [f"auprc_f{k}" for k in range(1, 9)] + \
    ["auprc_peakprob", "peak_precision", "peak_recall", "peak_f1", "nbpeak_precision",
     "nbpeak_recall", "nbpeak_f1", "rspec", "pearson", "rspec_cal_128bp", "fe_pearson_25bp",
     "fe_spearman_25bp"]
BOARD_COLS = (["method", "table", "track_set", "comparison_label", "assay_class", "n_tracks",
               "n_cells"] + [c for m in BOARD_METRICS for c in (m, f"n_{m}")]
              + [f"{m}_{s}" for m in CI_METRICS for s in ("ci_lo", "ci_hi")]
              + ["rspec_cal_seed_range", "n_seeds", "ag2_status", "ag2_api_label", "trained_on_track",
                 "lambda_source"]
              + LABEL_COLS)
CONTRAST_COLS = ["method", "reference", "table", "track_set", "comparison_label", "truth_peaks",
                 "assay_class", "metric", "n_tracks", "n_cells", "mean_method", "mean_reference",
                 "mean_diff", "ci_lo", "ci_hi", "n_ahead", "call", "preregistered"]
EXCLUDED_COLS = ["track_set", "assay_class", "truth_peaks", "metric", "target", "method", "value"]


class Refused(RuntimeError):
    pass


# ------------------------------------------------------------------------------------ helpers

def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _num(v) -> bool:
    return isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if _num(v):
        v = float(v)
        return "nan" if math.isnan(v) else f"{v:.6g}"
    return str(v)


def _write_tsv(path: Path, cols, rows) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(r.get(c, "")) for c in cols])
    tmp.replace(path)


def corr(x, y) -> float:
    return Mx._corr(np.asarray(x, np.float64), np.asarray(y, np.float64))


def spearman(p, y, nsub: int = NSUB) -> float:
    """Spearman on Max's fixed subsample (score_f1.py l.100-101, l.280): `nsub` bins drawn with
    seed 0 without replacement when the track is longer, else every bin."""
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    if p.size > nsub:
        sub = np.sort(np.random.default_rng(0).choice(p.size, size=nsub, replace=False))
        p, y = p[sub], y[sub]
    return corr(rankdata(p), rankdata(y))


def to_128(x, regions, rb: int) -> np.ndarray:
    """Average a [n_regions * rb] 25 bp track to the genome-aligned 128 bp bins that lie wholly
    inside each region (AG2's own grid: windows start on a multiple of 128 bp)."""
    x = np.asarray(x, np.float64)
    out = []
    for i, (_, sb) in enumerate(regions):
        s = int(sb) * BIN_BP
        e = s + rb * BIN_BP
        first = -(-s // AG2_BIN_BP) * AG2_BIN_BP
        n = (e - first) // AG2_BIN_BP
        out.append(SP.resample_mean(x[i * rb:(i + 1) * rb], s, BIN_BP, first, AG2_BIN_BP, n))
    return np.concatenate(out)


def cluster_bootstrap(values, clusters, n_boot: int = 2000, seed: int = 0):
    """95% interval of the mean of `values` under resampling of clusters with replacement
    (summarize_f1.py `boot_ci`, l.46-53: per resample, the sum over the drawn clusters over their
    count). NaN values are dropped first. Returns (lo, hi)."""
    v = np.asarray(values, np.float64)
    c = np.asarray(clusters, dtype=object)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    if v.size == 0:
        return float("nan"), float("nan")
    keys = sorted(set(c.tolist()))
    s = np.array([v[c == k].sum() for k in keys])
    n = np.array([(c == k).sum() for k in keys], np.float64)
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(int(n_boot), len(keys)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def _resolve(d: Path, regime: str):
    """(regime dir, tracks.json) of a method directory for `regime`."""
    d = Path(d)
    if (d / regime / "tracks.json").is_file():
        d = d / regime
    if not (d / "tracks.json").is_file():
        raise Refused(f"{d}: no tracks.json for regime {regime} (pinned interface F)")
    meta = json.loads((d / "tracks.json").read_text(encoding="utf-8"))
    if meta.get("kind") != "ag2" and meta.get("regime", regime) != regime:
        raise Refused(f"{d}: regime {meta.get('regime')!r}, not {regime!r}")
    if meta.get("kind") not in KINDS:
        raise Refused(f"{d}: kind {meta.get('kind')!r} not in {KINDS}")
    return d, meta


def _roles(meta: dict) -> set:
    kind = meta["kind"]
    if kind in NB_KINDS:
        r = meta.get("trained_on_roles")
        if r is None:
            if kind == "run":
                raise Refused(f"run {meta.get('method')!r}: tracks.json has no trained_on_roles")
            r = ["train"]
        return set(r.split(",") if isinstance(r, str) else r)
    if kind == "external":
        return {"train", "val"}
    return set()


def _trained(kind: str, roles: set, role: str, status: str) -> bool:
    if kind == "ag2":
        return status in AG2_TRAINED_STATUS
    return role in roles


def _per_target(v, mid):
    return v.get(mid) if isinstance(v, dict) else v


def lambda_and_control(meta: dict, mid: str, table: str) -> tuple[str, str]:
    """(lambda_source, uses_target_cell_control) of one target of one method directory, from its
    `tracks.json` (`lambda_source`: a string or {measurement_id: string}). In the control_free table
    a run or baseline reads `signal_pc`, whose source is `lambda_source_predicted_control` (default
    `predicted_control`). An external entry states `uses_target_cell_control` (yes|no) itself."""
    kind = meta["kind"]
    if kind in NB_KINDS and table == "control_free":
        src = _per_target(meta.get("lambda_source_predicted_control", "predicted_control"), mid)
    else:
        src = _per_target(meta.get("lambda_source"), mid)
    if kind == "external":
        ctl = _per_target(meta.get("uses_target_cell_control"), mid)
        if ctl not in ("yes", "no"):
            raise Refused(f"external {meta.get('method')!r}: tracks.json states no "
                          f"uses_target_cell_control (yes|no) for {mid}")
        return str(src or "external_entry"), ctl
    if src not in CONTROL_OF_LAMBDA:
        raise Refused(f"{meta.get('method')!r} ({kind}): lambda_source {src!r} of {mid} is not one "
                      f"of {sorted(CONTROL_OF_LAMBDA)}")
    return src, CONTROL_OF_LAMBDA[src]


def _labels(method: str, kind: str, ctl: str, truth_peaks: str) -> dict:
    reads = "no" if kind == "ag2" or method == MEAN_METHOD else "yes"
    return {"trained_on_cell_other_tracks": "yes",
            "reads_target_cell_inputs_at_test_positions": reads,
            "uses_target_cell_control": ctl, "truth_peaks": truth_peaks,
            "recalibration": f"oracle_isotonic_{NCAL}",
            "noise_floor_row": f"{HALF_METHOD} (half against half; a floor at half depth)"}


# ------------------------------------------------------------------------------- per target

def signal_metrics(p, y, a, regions, rb: int) -> tuple[dict, np.ndarray | None]:
    """Signal-space metrics of one prediction `p` against truth `y` with training average `a`.
    Returns (metrics, recalibrated prediction or None)."""
    p, y, a = (np.asarray(v, np.float64) for v in (p, y, a))
    if not (np.isfinite(p).all() and np.isfinite(y).all() and np.isfinite(a).all()):
        u = "undefined: non-finite values"
        return {k: u for k in ("rspec_cal", "pearson_cal", "spearman", "rspec", "pearson",
                               "rspec_cal_128bp")}, None
    cal = SP.isotonic_recalibrate(p, y, n_bins=NCAL)
    out = {"rspec_cal": float(Mx.r_spec(y, cal, a)), "pearson_cal": corr(cal, y),
           "spearman": spearman(p, y), "rspec": float(Mx.r_spec(y, p, a)), "pearson": corr(p, y)}
    out["rspec_cal_128bp"] = float(Mx.r_spec(to_128(y, regions, rb), to_128(cal, regions, rb),
                                             to_128(a, regions, rb)))
    return out, cal


def peak_metrics(target_tiles, cons, freq, score_tiles, called_tiles, prob_tiles) -> dict:
    """Peak metrics of one row on one truth source. `prob_tiles` is an array or a reason string."""
    out = {}
    if target_tiles is None:
        for k in PEAK_METRICS:
            out[k] = NA_AVAIL
        return out
    out["auprc"], out["auprc_posfrac"] = PK.auprc(target_tiles, cons, score_tiles)
    for b in PK.auprc_by_frequency(target_tiles, cons, score_tiles, freq):
        out[f"auprc_f{b['bin']}"] = b["auprc"]
    out["peak_precision"], out["peak_recall"], out["peak_f1"] = \
        PK.binary_prf(target_tiles, cons, called_tiles)
    if isinstance(prob_tiles, str):
        for k in ("auprc_peakprob", "nbpeak_precision", "nbpeak_recall", "nbpeak_f1"):
            out[k] = prob_tiles
    else:
        out["auprc_peakprob"] = PK.auprc(target_tiles, cons, prob_tiles)[0]
        out["nbpeak_precision"], out["nbpeak_recall"], out["nbpeak_f1"] = \
            PK.binary_prf(target_tiles, cons, prob_tiles >= PROB_CALL)
    return out


# ------------------------------------------------------------------------------------ score

def _parse_methods(specs) -> dict:
    out = {}
    for s in specs or []:
        if "=" not in s:
            raise Refused(f"--method {s!r}: expected <name>=<dir>[,<dir>...]")
        name, dirs = s.split("=", 1)
        if name in out or name in (AG2_MATCHED, AG2_OTHER):
            raise Refused(f"--method {name}: given twice or a reserved AG2 name")
        out[name] = [Path(x) for x in dirs.split(",") if x]
    return out


def score(evalset, audit, tables, truth_from, methods, ag2, peaks, out, *, split="val",
          allow_blind=False, repeat_blind_reason=None, table="headline", n_boot=2000,
          allow_unscored=False, log=print) -> dict:
    if split not in ("val", "blind"):
        raise ValueError(f"split {split!r} not in (val, blind)")
    if table not in TABLE_REGIME:
        raise ValueError(f"table {table!r} not in {tuple(TABLE_REGIME)}")
    if split == "blind" and not allow_blind:
        raise PermissionError("--split blind needs --allow-blind (gate G4)")
    if split == "blind" and table != "headline":
        raise Refused("the blind pass has the headline table only (decision 11)")
    regime = TABLE_REGIME[table]
    evalset, tables, out = Path(evalset).resolve(), Path(tables), Path(out)
    ag2 = Path(ag2)
    ev = json.loads(evalset.read_text(encoding="utf-8"))
    esha = E.sha256_file(evalset)
    rb = int(ev.get("region_bins", 2 ** 14))
    regions = [(str(c), int(s)) for c, s in ev["regions"]]
    rsha = regions_sha256(regions, rb)
    ev_targets = {t["target"]: t for t in ev.get("imputation_targets", [])
                  if t.get("split", "val") == split and t["regime"] == regime}

    # --- method directories: resolve and check before any data read
    specs = _parse_methods(methods)
    if MEAN_METHOD not in specs:
        raise Refused(f"--method {MEAN_METHOD}=<dir> is required (the `a` of r_spec)")
    tdir, tmeta = _resolve(Path(truth_from), regime)
    if tmeta["kind"] not in NB_KINDS:
        raise Refused(f"{tdir}: kind {tmeta['kind']!r}; the truth comes from a run or baseline")
    adir, ameta = _resolve(ag2, regime)
    if ameta["kind"] != "ag2":
        raise Refused(f"{adir}: kind {ameta['kind']!r}, not ag2")
    if not (adir / "EVALUATION_ONLY").is_file():
        raise Refused(f"{adir}: no EVALUATION_ONLY marker (decision 12)")
    dirs = {name: [_resolve(d, regime) for d in ds] for name, ds in specs.items()}
    dirs[AG2_MATCHED] = dirs[AG2_OTHER] = [(adir, ameta)]
    for name, ds in list(dirs.items()) + [("truth-from", [(tdir, tmeta)])]:
        for d, m in ds:
            if m.get("regions_sha256") != rsha or int(m.get("region_bins", -1)) != rb:
                raise Refused(f"{name} ({d}): regions_sha256 or region_bins differ from the eval "
                              f"set's ({rsha[:12]}, {rb})")
            if m.get("split") != split:
                raise Refused(f"{name} ({d}): split {m.get('split')!r}, not {split!r}")
        for _, m in ds:
            _roles(m)                                  # a run must state trained_on_roles
        kinds = {m["kind"] for _, m in ds}
        if len(kinds) > 1:
            raise Refused(f"{name}: its seed directories differ in kind {sorted(kinds)}")

    # --- targets, audit, track sets
    rows = E.read_tsv(tables / "measurements.tsv")
    by_id = {r["measurement_id"]: r for r in rows}
    aud = defaultdict(list)
    for r in E.read_tsv(audit):
        aud[r["eic_track"]].append(r)
    universe = sorted(t for t in tmeta.get("targets", []) if t in ev_targets)
    info, unscored = {}, []
    for mid in sorted(set(ev_targets) - set(tmeta.get("targets", []))):
        unscored.append({"target": mid, "cell": ev_targets[mid].get("cell", ""),
                         "assay": ev_targets[mid].get("assay", ""), "role": split,
                         "reason": f"no truth in --truth-from {tdir}"})
    if unscored and not allow_unscored:
        raise Refused(f"--truth-from {tdir} lacks {len(unscored)} {split} target(s) of regime "
                      f"{regime} in the eval set, e.g. {unscored[0]['target']} "
                      f"(--allow-unscored lists them in unscored_targets.tsv)")
    for mid in universe:
        row = by_id.get(mid)
        if row is None:
            raise Refused(f"{mid}: not in {tables / 'measurements.tsv'}")
        ar = aud.get(row["eic_track"])
        if not ar:
            raise Refused(f"{mid} ({row['eic_track']}): no row in the audit {audit}")
        sets = {SET_OF_STATUS.get(r["status"], NO_SET) for r in ar}
        if len(sets) > 1:
            raise Refused(f"{mid} ({row['eic_track']}): listed in two track sets {sorted(sets)}")
        st = ar[0]["status"]
        info[mid] = {"target": mid, "eic_track": row["eic_track"], "cell": row["cell"],
                     "assay": row["assay"], "assay_class": row.get("assay_class", ""),
                     "role": ev_targets[mid].get("split", split), "status": st,
                     "api_label": ar[0].get("api_label", ""), "set": sets.pop()}
        if info[mid]["set"] == NO_SET:
            unscored.append(dict(info[mid], reason=f"status {st}: in no comparison set"))
    scored = [m for m in universe if info[m]["set"] != NO_SET]
    for name, ds in dirs.items():
        for d, m in ds:
            if m["kind"] == "replicate_floor":
                continue
            miss = [t for t in scored if t not in set(m.get("targets", []))
                    and not (m["kind"] == "ag2" and (d / f"{t}.matched.signal.npy").is_file())]
            if miss:
                raise Refused(f"{name} ({d}): lacks {len(miss)} scored target(s), e.g. {miss[0]}")

    # --- blind ledger: started before the first data read
    ledger, entry = None, None
    if split == "blind":
        from minicandi.dataset import tables_sha256          # lazy: imports torch
        from onion_eval import score as S                     # lazy: imports torch
        ledger = evalset.parent / S.BLIND_LEDGER
        key = f"{LEDGER_PREFIX}{ag2.resolve()}"
        prior = [r for r in S.ledger_rows(ledger)
                 if r["run"] == key and r["evalset_sha256"] == esha]
        reason = (repeat_blind_reason or "").strip()
        if prior and not reason:
            raise Refused(f"{key} already had {len(prior)} blind-pass ledger rows for this eval set "
                          f"({ledger}); a repeat needs --repeat-blind-reason \"<text>\"")
        entry = {"run": key, "evalset_sha256": esha, "tables_sha256": tables_sha256(tables),
                 "git_sha": S.git_sha(), "out": str(out.resolve()),
                 "repeat_reason": reason if prior else ""}
        S.ledger_append(ledger, dict(entry, time=_utc(), status="started"))

    # --- truth, and the equal-truth check
    truth = {}
    for mid in scored:
        y = np.load(tdir / f"{mid}.truth.npy").astype(np.float64)
        if y.shape != (len(regions) * rb,):
            raise Refused(f"{tdir}/{mid}.truth.npy: shape {y.shape}")
        truth[mid] = y
    for name, ds in dirs.items():
        for d, m in ds:
            if m["kind"] == "replicate_floor":
                continue
            for mid in scored:
                f = d / f"{mid}.truth.npy"
                if f.is_file() and not np.array_equal(np.load(f).astype(np.float64), truth[mid]):
                    raise Refused(f"{name} ({d}): truth of {mid} differs from {tdir}")

    # --- tile sets per assay and truth source
    tiles = _tile_sets(rows, tables, peaks, split, scored, info, truth, regions, rb, log)

    # --- per target
    channels = {}
    if (adir / "channels.tsv").is_file():
        channels = {r["measurement_id"]: r for r in E.read_tsv(adir / "channels.tsv")}
    per = []
    for mid in scored:
        t = info[mid]
        y = truth[mid]
        a = np.load(dirs[MEAN_METHOD][0][0] / f"{mid}.signal.npy").astype(np.float64)
        nat_truth_f = tdir / f"{mid}.native_truth.npy"
        nat_truth = np.load(nat_truth_f).astype(np.float64) if nat_truth_f.is_file() else None
        for name, ds in dirs.items():
            kind = ds[0][1]["kind"]
            roles = _roles(ds[0][1])
            trained = _trained(kind, roles, t["role"], t["status"])
            tset = t["set"] if (kind == "ag2" or not trained) else NO_SET
            seed_vals = [_score_one(name, kind, d, mid, y, a, nat_truth, regions, rb, table,
                                    {tp: tiles[(t["assay"], tp)] for tp in TRUTH_PEAKS})
                         for d, _ in ds]
            api = t["api_label"]
            if kind == "ag2" and mid in channels:
                api = channels[mid].get("api_label_matched" if name == AG2_MATCHED
                                        else "api_label_otherend", api) or api
            lc = {lambda_and_control(m, mid, table) for _, m in ds}
            if len(lc) > 1:
                raise Refused(f"{name}: seed directories differ in lambda_source for {mid}: {sorted(lc)}")
            lam, ctl = lc.pop()
            for tp in TRUTH_PEAKS:
                rec = _mean_seeds([sv[tp] for sv in seed_vals])
                rec.update(t, method=name, kind=kind, table=table, regime=regime, api_label=api,
                           track_set=tset, n_seeds=len(ds),
                           comparison_label=COMPARISON_LABEL.get(tset, "not in a comparison set"),
                           trained_on_track="yes" if trained else "no", lambda_source=lam,
                           **_labels(name, kind, ctl, tp))
                per.append(rec)
    out.mkdir(parents=True, exist_ok=True)
    _write_tsv(out / "per_target.tsv", PER_TARGET_COLS, per)
    _write_tsv(out / "unscored_targets.tsv",
               ["target", "eic_track", "cell", "assay", "role", "status", "reason"], unscored)
    board, excluded = _board(per, table, n_boot)
    _write_tsv(out / f"ag2_board_{split}.tsv", BOARD_COLS, board)
    _write_tsv(out / f"ag2_board_excluded_{split}.tsv", EXCLUDED_COLS, excluded)
    con = _contrasts(per, table, n_boot)
    _write_tsv(out / f"ag2_contrasts_{split}.tsv", CONTRAST_COLS, con)
    st = Counter(info[m]["status"] for m in universe)
    summary = {"split": split, "table": table, "regime": regime, "n_targets": len(universe),
               "n_scored": len(scored), "n_unscored": len(unscored), "status_counts": dict(sorted(st.items())),
               "sets": dict(Counter(info[m]["set"] for m in scored)),
               "methods": {n: [{"dir": str(d.resolve()), "kind": m["kind"],
                                "evalset_sha256": m.get("evalset_sha256", "")} for d, m in ds]
                           for n, ds in dirs.items()},
               "truth_from": str(tdir.resolve()), "evalset_sha256": esha, "time": _utc()}
    (out / "SCORED").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                encoding="utf-8")
    log(f"{split} {table}: {len(scored)} of {len(universe)} targets scored; statuses "
        + ", ".join(f"{k} {v}" for k, v in sorted(st.items())))
    if ledger is not None:
        from onion_eval import score as S
        S.ledger_append(ledger, dict(entry, time=_utc(), status="scored"))
    return summary


def _tile_sets(rows, tables, peaks, split, scored, info, truth, regions, rb, log) -> dict:
    """(assay, truth source) -> {"cons": bool tiles, "freq": float tiles, <mid>: target tiles}."""
    roles = ("train", "val", "blind") if split == "blind" else ("train", "val")
    second = E.second_experiment_ids(tables)
    keep = [r for r in rows if r["measurement_id"] not in second]
    views = E.base_views(keep, roles)
    peaks = Path(peaks)
    raster = {}

    def peak_tiles(eic):
        if eic not in raster:
            f = next((peaks / f"{eic}{s}" for s in (".narrowPeak.gz", ".narrowPeak")
                      if (peaks / f"{eic}{s}").is_file()), None)
            raster[eic] = None if f is None else PK.tile_any(PK.rasterize(f, regions, rb),
                                                             region_bins=rb)
        return raster[eic]

    out = {}
    for assay in sorted({info[m]["assay"] for m in scored}):
        by_cell = {}
        for (cell, a), r in views.items():
            if a != assay:
                continue
            tl = peak_tiles(r["eic_track"])
            if tl is not None:
                by_cell[cell] = tl if cell not in by_cell else (by_cell[cell] | tl)
        cons, freq = PK.consensus(by_cell) if by_cell else (None, None)
        enc = {"cons": cons, "freq": freq}
        for m in scored:
            if info[m]["assay"] == assay:
                tl = peak_tiles(info[m]["eic_track"])
                enc[m] = tl if (tl is not None and cons is not None) else None
        out[(assay, "encode")] = enc
        der_cells = {}
        der = {}
        for m in scored:
            if info[m]["assay"] != assay:
                continue
            tl = PK.tile_any(truth[m] >= PEAK_CALL, region_bins=rb)
            der[m] = tl
            c = info[m]["cell"]
            der_cells[c] = tl if c not in der_cells else (der_cells[c] | tl)
        der["cons"], der["freq"] = PK.consensus(der_cells)
        out[(assay, "pipeline_derived")] = der
        log(f"{assay}: encode tile set {len(by_cell)} cells, "
            f"{0 if cons is None else int(cons.sum())} consensus tiles; derived tile set "
            f"{len(der_cells)} cells, {int(der['cons'].sum())} consensus tiles")
    return out


def _score_one(name, kind, d, mid, y, a, nat_truth, regions, rb, table, ts_by_source):
    """Metrics of one method directory on one target: {truth source: metrics}. `ts_by_source`
    maps a truth source to its tile set ({"cons", "freq", <mid>: target tiles or None})."""
    if kind == "ag2":
        f = d / f"{mid}.{'matched' if name == AG2_MATCHED else 'otherend'}.signal.npy"
    elif kind in NB_KINDS and table == "control_free":
        f = d / f"{mid}.signal_pc.npy"
    else:
        f = d / f"{mid}.signal.npy"
    if not f.is_file():
        return {tp: {k: NA_AVAIL for k in METRICS} for tp in ts_by_source}
    p = np.load(f).astype(np.float64)
    if p.shape != y.shape:
        raise Refused(f"{f}: shape {p.shape}, expected {y.shape}")
    yy = y
    if kind == "replicate_floor":                  # half against half
        yy = np.load(d / f"{mid}.truth.npy").astype(np.float64)
    rec, cal = signal_metrics(p, yy, a, regions, rb)
    if name == MEAN_METHOD and cal is not None:    # a function of `a` scores 0 (score_f1.py l.282-283)
        rec.update(rspec=0.0, rspec_cal=0.0, rspec_cal_128bp=0.0)
    # fold change / insertion space (secondary A or B)
    if kind not in NB_KINDS and kind != "ag2":
        rec["fe_pearson_25bp"] = rec["fe_spearman_25bp"] = NA_APPL
    elif table != "headline" or nat_truth is None:
        rec["fe_pearson_25bp"] = rec["fe_spearman_25bp"] = NA_AVAIL
    else:
        nf = f if kind == "ag2" else d / f"{mid}.native.npy"
        if nf.is_file():
            nat = np.load(nf).astype(np.float64)
            rec["fe_pearson_25bp"] = corr(nat, nat_truth)
            rec["fe_spearman_25bp"] = spearman(nat, nat_truth)
        else:
            rec["fe_pearson_25bp"] = rec["fe_spearman_25bp"] = NA_AVAIL
    # peaks
    if kind not in NB_KINDS:
        prob = NA_APPL
    elif table != "headline" or not (d / f"{mid}.peakprob.npy").is_file():
        prob = NA_AVAIL
    else:
        prob = PK.tile_max(np.load(d / f"{mid}.peakprob.npy"), region_bins=rb)
    if cal is None:
        rec.update({k: "undefined: non-finite values" for k in PEAK_METRICS})
        return {tp: dict(rec) for tp in ts_by_source}
    score_tiles = PK.tile_mean(p, region_bins=rb)
    called = PK.tile_any(cal >= PEAK_CALL, region_bins=rb)
    return {tp: dict(rec, **peak_metrics(ts.get(mid), ts["cons"], ts["freq"], score_tiles, called,
                                         prob))
            for tp, ts in ts_by_source.items()}


def _mean_seeds(vals: list[dict]) -> dict:
    out = {}
    for k in METRICS:
        xs = [v[k] for v in vals]
        if all(_num(x) for x in xs):
            out[k] = float(np.mean(xs))
        else:
            out[k] = next(x for x in xs if not _num(x))
    rc = [v["rspec_cal"] for v in vals]
    out["rspec_cal_seed_range"] = (float(max(rc) - min(rc)) if len(rc) > 1 and all(_num(x) for x in rc)
                                   else NA_APPL)
    return out


# ---------------------------------------------------------------------------- board, contrasts

def _agg(xs):
    nums = [float(x) for x in xs if _num(x)]
    if not nums:
        strs = [x for x in xs if isinstance(x, str)]
        return strs[0] if strs else float("nan")
    a = np.asarray(nums)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def _counts(xs) -> str:
    return "; ".join(f"{k}: {v}" for k, v in sorted(Counter(xs).items()))


def _enough(n_tracks: int, n_cells: int) -> bool:
    return n_tracks >= MIN_TRACKS and n_cells >= MIN_CELLS


def _finite(v) -> bool:
    return _num(v) and math.isfinite(float(v))


def _yes_count(xs) -> str:
    xs = list(xs)
    ny = sum(x == "yes" for x in xs)
    return "no" if ny == 0 else "yes" if ny == len(xs) else f"yes: {ny} of {len(xs)}"


def _board(per, table, n_boot) -> tuple[list, list]:
    """Board rows and the excluded (track, method, metric) cases. Every mean of a (track set, assay
    class, truth source, metric) is taken on the common tracks: those where every method of the
    board that has a number for the metric there has a finite value. `n_<metric>` is that count. A
    method with no number at all for the metric prints its reason (`not applicable`, ...)."""
    groups = defaultdict(lambda: defaultdict(dict))
    for r in per:
        if r["track_set"] == NO_SET:
            continue
        for ac in ("all", r["assay_class"]):
            groups[(r["track_set"], ac, r["truth_peaks"])][r["method"]][r["target"]] = r
    out, excluded = [], []
    for (tset, ac, tp), by_m in sorted(groups.items()):
        targets = sorted({t for rs in by_m.values() for t in rs})
        common = {}
        for k in BOARD_METRICS:
            need = ("auprc", "auprc_posfrac") if k == "auprc_weighted" else (k,)
            part = [m for m, rs in by_m.items() if any(_num(r[need[0]]) for r in rs.values())]
            keep = [t for t in targets
                    if all(t in by_m[m] and all(_finite(by_m[m][t][q]) for q in need) for m in part)]
            common[k] = (part, keep)
            if k == "auprc_weighted":
                continue
            for t in sorted(set(targets) - set(keep)):
                for m in part:
                    v = by_m[m][t][k] if t in by_m[m] else "no row"
                    if not _finite(v):
                        excluded.append({"track_set": tset, "assay_class": ac, "truth_peaks": tp,
                                         "metric": k, "target": t, "method": m, "value": v})
        for method, rmap in sorted(by_m.items()):
            rs = list(rmap.values())
            b = {"method": method, "table": table, "track_set": tset,
                 "comparison_label": COMPARISON_LABEL[tset], "assay_class": ac,
                 "n_tracks": len(rs), "n_cells": len({r["cell"] for r in rs}),
                 "n_seeds": rs[0]["n_seeds"]}
            for k in BOARD_METRICS:
                part, keep = common[k]
                base = "auprc" if k == "auprc_weighted" else k
                if method not in part:
                    b[k] = _agg([r[base] for r in rs])
                    b[f"n_{k}"] = 0
                    continue
                b[f"n_{k}"] = len(keep)
                if not keep:
                    b[k] = float("nan")
                elif k == "auprc_weighted":
                    x = np.array([float(rmap[t]["auprc"]) for t in keep])
                    w = np.array([float(rmap[t]["auprc_posfrac"]) for t in keep])
                    b[k] = float((x * w).sum() / w.sum()) if w.sum() > 0 else float("nan")
                else:
                    b[k] = float(np.mean([float(rmap[t][k]) for t in keep]))
                if k in CI_METRICS:
                    cells = [rmap[t]["cell"] for t in keep]
                    if not _enough(len(keep), len(set(cells))):
                        b[f"{k}_ci_lo"] = b[f"{k}_ci_hi"] = TOO_FEW
                    else:
                        b[f"{k}_ci_lo"], b[f"{k}_ci_hi"] = cluster_bootstrap(
                            [float(rmap[t][k]) for t in keep], cells, n_boot)
            for k in CI_METRICS:
                if f"{k}_ci_lo" not in b:
                    b[f"{k}_ci_lo"] = b[f"{k}_ci_hi"] = b[k] if isinstance(b[k], str) else "nan"
            b["rspec_cal_seed_range"] = _agg([r["rspec_cal_seed_range"] for r in rs])
            b["ag2_status"] = _counts(r["status"] for r in rs)
            b["ag2_api_label"] = _counts(r["api_label"] for r in rs)
            b["trained_on_track"] = _yes_count(r["trained_on_track"] for r in rs)
            b["lambda_source"] = _counts(r["lambda_source"] for r in rs)
            for k in LABEL_COLS:
                b[k] = rs[0][k]
            b["uses_target_cell_control"] = _yes_count(r["uses_target_cell_control"] for r in rs)
            out.append(b)
    return out, excluded


def _contrasts(per, table, n_boot) -> list:
    idx = {(r["method"], r["target"], r["truth_peaks"]): r for r in per}
    methods = sorted({r["method"] for r in per if r["kind"] != "replicate_floor"} - {AG2_MATCHED})
    out = []
    for tset in (UNSEEN, AG2_TRAINED):            # never pooled
        for method in methods:
            for metric in CONTRAST_METRICS:
                tps = TRUTH_PEAKS if metric.startswith("auprc") else ("encode",)
                for tp in tps:
                    pairs = []
                    for (m, t, p), r in idx.items():
                        if m != method or p != tp or r["track_set"] != tset:
                            continue
                        ref = idx.get((AG2_MATCHED, t, tp))
                        if ref is None or ref["track_set"] != tset:
                            continue
                        x, z = r[metric], ref[metric]
                        if _num(x) and _num(z) and math.isfinite(x) and math.isfinite(z):
                            pairs.append((r["assay_class"], r["cell"], float(x), float(z)))
                    for ac in ["all"] + sorted({q[0] for q in pairs}):
                        sel = [q for q in pairs if ac == "all" or q[0] == ac]
                        if not sel:
                            continue
                        d = np.array([x - z for _, _, x, z in sel])
                        cells = [c for _, c, _, _ in sel]
                        row = {"method": method, "reference": AG2_MATCHED, "table": table,
                               "track_set": tset, "comparison_label": COMPARISON_LABEL[tset],
                               "truth_peaks": tp if metric.startswith("auprc") else "not used",
                               "assay_class": ac, "metric": metric, "n_tracks": len(sel),
                               "n_cells": len(set(cells)),
                               "mean_method": float(np.mean([x for _, _, x, _ in sel])),
                               "mean_reference": float(np.mean([z for _, _, _, z in sel])),
                               "mean_diff": float(d.mean()), "n_ahead": int((d > 0).sum()),
                               "preregistered": int((method, AG2_MATCHED, metric, tset) == PREREGISTERED
                                                     and ac == "all")}
                        if _enough(len(sel), len(set(cells))):
                            lo, hi = cluster_bootstrap(d, cells, n_boot)
                            row.update(ci_lo=lo, ci_hi=hi,
                                       call=(f"{method} ahead" if lo > 0 else
                                             f"{AG2_MATCHED} ahead" if hi < 0 else "no call"))
                        else:
                            row.update(ci_lo=TOO_FEW, ci_hi=TOO_FEW, call=TOO_FEW)
                        out.append(row)
    return out


# ------------------------------------------------------------------------------------- gate

GATE_TOL = {"pearson_cal": 0.005, "spearman": 0.005, "auprc": 0.005, "rspec_cal": 0.01}
TILE = 20


def _term_of(b: str) -> str:
    import re
    return re.sub(r"_(nonrep|grp\d+_rep\d+)$", "", b)


def _to25(v128, n25):
    """score_f1.py `to25` (l.210-214) through `ag2_space.resample_mean` (bins from 0 bp)."""
    v = SP.resample_mean(np.asarray(v128, np.float64), 0, AG2_BIN_BP, 0, BIN_BP, n25)
    return np.nan_to_num(v, nan=0.0)


def gate(targets, pred, store, tiles, reference, out, *, tracks=None,
         chroms=("chr20", "chr21", "chr22"), store_obj=None, log=print) -> bool:
    """Reproduction gate (section B.6): score Max's cached AG2 matched channel against his truth with
    this module's functions, on his geometry (score_f1.py l.81-101: genome-mask bins, whole valid
    tiles, the seed-0 subsample), and compare with his per-target table (method `ag2`, setting
    `all`, stratum `all`). Returns True on PASS. FAIL when any target of `--targets` is skipped
    (no reference row, no prediction, no peak tiles), when one side of a metric is NaN and the other
    is not, when `rspec_cal` cannot be computed (no `--tracks`), or when a difference passes its
    tolerance. Skipped targets go to `gate_skipped.tsv`."""
    if store_obj is None:
        from candi.store.reader import CorpusStore
        store_obj = CorpusStore(store)
    C, CS = store_obj, list(chroms)
    pred, tiles, out = Path(pred), Path(tiles), Path(out)
    tg = E.read_tsv(targets)
    ref = {(r["biosample"], r["assay"]): r for r in E.read_tsv(reference)
           if r.get("method") == "ag2" and r.get("setting", "all") == "all"
           and r.get("stratum", "all") == "all"}
    tr = E.read_tsv(tracks) if tracks else None
    mask = {c: np.asarray(C.genome.mask(c, 0, C.n_bins(c)), bool) for c in CS}
    tl, toff = [], 0
    for c in CS:
        g = np.flatnonzero(mask[c])
        nt = C.n_bins(c) // TILE
        t = g // TILE
        tl.append(np.where(t < nt, toff + t, -1))
        toff += nt
    tile_of = np.concatenate(tl)
    NT = toff
    tidx = np.where(tile_of >= 0, tile_of, NT)
    tile_valid = np.bincount(tidx, minlength=NT + 1)[:NT] == TILE

    def read_truth(b, a):
        return np.concatenate([np.arcsinh(np.asarray(C[b].pval(c, 0, C.n_bins(c), assays=[a]),
                                                     np.float64))[mask[c], 0] for c in CS])

    def calls_of(b, a):
        f = tiles / f"{b}.npz"
        if not f.is_file():
            return None
        z = np.load(f)
        names = [str(v) for v in z["tracks"]]
        if a not in names:
            return None
        m = np.concatenate([z[f"cnt_{c}"][:, names.index(a)] for c in CS]) > 0
        return m if m[tile_valid].any() else None

    rows, skipped, compared = [], [], set()
    for a in sorted({r["assay"] for r in tg}):
        if tr is not None:
            bios = sorted({r["biosample"] for r in tr if r["assay"] == a})
        else:
            bios = sorted(p.stem for p in tiles.glob("*.npz"))
        by_term = {}
        calls = {}
        for b in bios:
            m = calls_of(b, a)
            if m is not None:
                calls[b] = m
                k = _term_of(b)
                by_term[k] = m.copy() if k not in by_term else (by_term[k] | m)
        cnt = np.zeros(NT, np.int64)
        for m in by_term.values():
            cnt += m
        cons = tile_valid & (cnt > 0)
        avg = None
        if tr is not None:
            train, seen = [], set()
            for r in sorted((r for r in tr if r["assay"] == a and r.get("hidden") == "False"),
                            key=lambda r: r["biosample"]):
                if r["exp_accession"] not in seen:
                    seen.add(r["exp_accession"])
                    train.append(r["biosample"])
            if train:
                avg = np.mean([read_truth(b, a) for b in train], axis=0)
        for r in (r for r in tg if r["assay"] == a):
            b = r["biosample"]
            files = [pred / c / f"{b}__{a}.npz" for c in CS]
            why = ("no reference row" if (b, a) not in ref else
                   "no prediction file" if not all(f.is_file() for f in files) else
                   "no peak tiles" if b not in calls else "")
            if why:
                skipped.append({"biosample": b, "assay": a, "reason": why})
                continue
            compared.add((b, a))
            p = np.concatenate([_to25(np.load(f)["matched"], C.n_bins(c))[mask[c]]
                                for f, c in zip(files, CS)])
            y = read_truth(b, a)
            cal = SP.isotonic_recalibrate(p, y, n_bins=NCAL)
            s = np.bincount(tidx, weights=p, minlength=NT + 1)[:NT] / TILE
            got = {"pearson_cal": corr(cal, y), "spearman": spearman(p, y),
                   "auprc": PK.auprc(calls[b], cons, s)[0],
                   "rspec_cal": float(Mx.r_spec(y, cal, avg)) if avg is not None else NA_AVAIL}
            for k, v in got.items():
                theirs = float(ref[(b, a)][k]) if ref[(b, a)].get(k, "") not in ("", "nan") \
                    else float("nan")
                if not _num(v):
                    status, diff = f"ours {v}", float("nan")
                elif math.isnan(v) and math.isnan(theirs):
                    status, diff = "both undefined", 0.0
                elif math.isnan(v) or math.isnan(theirs):
                    status, diff = "one side undefined", float("nan")
                else:
                    status, diff = "compared", abs(v - theirs)
                rows.append({"biosample": b, "assay": a, "metric": k, "ours": v, "max": theirs,
                             "abs_diff": diff, "status": status})
    out.mkdir(parents=True, exist_ok=True)
    _write_tsv(out / "gate.tsv", ["biosample", "assay", "metric", "ours", "max", "abs_diff",
                                  "status"], rows)
    _write_tsv(out / "gate_skipped.tsv", ["biosample", "assay", "reason"], skipped)
    n_req = len({(r["biosample"], r["assay"]) for r in tg})
    log(f"targets compared: {len(compared)} of {n_req} requested"
        + (f"; skipped {len(skipped)} (gate_skipped.tsv)" if skipped else ""))
    ok = n_req > 0 and len(compared) == n_req
    for k, tol in GATE_TOL.items():
        rk = [r for r in rows if r["metric"] == k]
        bad = [r for r in rk if r["status"] not in ("compared", "both undefined")]
        d = [r["abs_diff"] for r in rk if r["status"] in ("compared", "both undefined")]
        if bad or not d:
            why = bad[0]["status"] if bad else "no value"
            log(f"{k}: {len(bad)} target(s) not comparable ({why})")
            ok = False
            continue
        mx = max(d)
        log(f"{k}: max |diff| {mx:.6g} over {len(d)} targets, tolerance {tol}")
        ok = ok and mx <= tol
    log(f"gate: {'PASS' if ok else 'FAIL'}")
    return ok


# -------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--evalset", required=True)
    s.add_argument("--audit", required=True)
    s.add_argument("--tables", required=True)
    s.add_argument("--truth-from", required=True)
    s.add_argument("--method", action="append", default=[])
    s.add_argument("--ag2", required=True)
    s.add_argument("--peaks", required=True)
    s.add_argument("--split", choices=("val", "blind"), required=True)
    s.add_argument("--allow-blind", action="store_true")
    s.add_argument("--repeat-blind-reason", default=None)
    s.add_argument("--table", choices=tuple(TABLE_REGIME), default="headline")
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--allow-unscored", action="store_true")
    s.add_argument("--out", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--targets", required=True)
    g.add_argument("--pred", required=True)
    g.add_argument("--store", required=True)
    g.add_argument("--tiles", required=True)
    g.add_argument("--reference", required=True)
    g.add_argument("--tracks", default=None)
    g.add_argument("--chroms", default="chr20,chr21,chr22")
    g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "score":
            score(a.evalset, a.audit, a.tables, a.truth_from, a.method, a.ag2, a.peaks, a.out,
                  split=a.split, allow_blind=a.allow_blind,
                  repeat_blind_reason=a.repeat_blind_reason, table=a.table, n_boot=a.n_boot,
                  allow_unscored=a.allow_unscored)
            return 0
        return 0 if gate(a.targets, a.pred, a.store, a.tiles, a.reference, a.out,
                         tracks=a.tracks, chroms=tuple(a.chroms.split(","))) else 1
    except (Refused, PermissionError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
