"""AlphaGenome 2 metadata tables and API smoke (t152 programme, Amendment 5, chunk AG0).

AlphaGenome 2 (AG2) outputs are evaluation-only in this run: never an input, a training target,
a baseline input or a selection signal for CANDI. This tool writes no prediction array.

    python tools/onion_data/ag2_meta.py smoke    --out DIR [--condition C] [--max-calls N]
    python tools/onion_data/ag2_meta.py metadata --out DIR [--channels TSV]

`smoke` makes at most 10 prediction calls (default 6) on one condition and writes
`<out>/API_SMOKE.md` (the four open API items, the client version and the observed metadata
shapes) and `<out>/prediction_channels.tsv` (every channel that a prediction returns, in the
column order of the returned arrays). `metadata` makes metadata calls only (free:
`observed_metadata()` and `supported_conditions()`) and writes the three tables of pinned
interface G plus `META_SCHEMA.md`:

    ag2_observed.tsv    one row per observed (condition, channel), human, CHIP_HISTONE/DNASE/ATAC
    ag2_channels.tsv    every channel of the three output types (from prediction_channels.tsv)
    ag2_conditions.tsv  every human reference condition that the API supports

`channel_index` is the column of the channel in the array that a prediction call returns; it is
read from `prediction_channels.tsv` (default `<out>/prediction_channels.tsv`), so `smoke` runs
before `metadata`. "Channel" is the API track name without its leading "<condition>_" prefix,
e.g. `Histone ChIP-seq-encode-H3K27ac-single-ended`.

Key rule (Amendment 5 decision 14): the key is read only from the file named by `AG2_KEY_FILE`
(default `~/.secrets/alphagenome_api_key`); the file must have mode 600 and hold exactly one
non-empty line. No key is taken as an argument, printed or written. API errors are logged by
their status code name only. `alphagenome` is imported inside the functions that call the API,
never at module level, so the unit tests run without it.

Run form (Nibi login node; compute nodes have no internet):

    module load gcc arrow/25.0.0 python/3.11 && \
      source /project/def-maxwl/mforooz/ONION/envs/ag2/bin/activate && \
      python tools/onion_data/ag2_meta.py metadata --out /project/def-maxwl/mforooz/ONION/ag2/meta
"""
from __future__ import annotations

import argparse
import os
import re
import stat
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_KEY_FILE = "~/.secrets/alphagenome_api_key"
OUTPUT_TYPES = ("CHIP_HISTONE", "DNASE", "ATAC")
ATTR = {"CHIP_HISTONE": "chip_histone", "DNASE": "dnase", "ATAC": "atac"}
ENCODE_DATA_SOURCE = "encode"
CURIE_RE = re.compile(r"^[A-Za-z]+:[0-9A-Za-z]+$")
MAX_SMOKE_CALLS = 10
# Shapes of observed_metadata(HOMO_SAPIENS) seen by the smoke of 2026-11-12.
SHAPES_SEEN_2026_11_12 = {"chip_histone": (1183, 12), "dnase": (385, 11), "atac": (179, 11)}
# Smoke default: K562 (cell line with observed histone, DNase-seq and ATAC-seq tracks), and 1 Mb
# windows on chr19 (2^20 bp, the length that AG2 was trained on).
SMOKE_CONDITION = "EFO:0002067"
SMOKE_WINDOW = 1_048_576
SMOKE_STARTS = tuple(range(5_000_000, 55_000_000, 5_000_000))  # 10 windows, one per call

OBSERVED_COLUMNS = ["output_type", "condition", "biosample_curie", "biosample_name", "target",
                    "endedness", "channel", "channel_index", "data_source", "n_observed_tracks"]
CHANNEL_COLUMNS = ["output_type", "channel_index", "channel", "target", "endedness"]
CONDITION_COLUMNS = ["condition", "biosample_curies", "biosample_name", "n_observed_encode_bulk",
                     "has_observed_accessibility"]


# --------------------------------------------------------------------------- key and client

def read_key(path: str | os.PathLike | None = None) -> str:
    """Return the API key from the key file (decision 14). Never print the result."""
    p = Path(os.path.expanduser(str(path or os.environ.get("AG2_KEY_FILE") or DEFAULT_KEY_FILE)))
    mode = stat.S_IMODE(p.stat().st_mode)
    if mode != 0o600:
        raise SystemExit(f"ag2_meta: key file {p} has mode {mode:o}, not 600; refused")
    lines = p.read_text().splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    if len(lines) != 1 or len(nonempty) != 1 or nonempty[0] != nonempty[0].strip():
        raise SystemExit(f"ag2_meta: key file {p} must hold exactly one non-empty line "
                         "with no surrounding blanks; refused")
    return nonempty[0]


def load_api():
    """Import the AG2 client lazily; returns a namespace of what this tool uses."""
    import types

    import alphagenome  # noqa: F401  (lazy: never at module level)
    from alphagenome.data import genome
    from alphagenome.models.v2 import dna_client
    return types.SimpleNamespace(
        version=getattr(alphagenome, "__version__", "unknown"),
        OutputType=dna_client.OutputType, Organism=dna_client.Organism,
        Interval=genome.Interval, create=dna_client.create)


def make_client(api):
    key = read_key()
    try:
        return api.create(key)
    finally:
        del key


def error_name(exc: BaseException) -> str:
    """Status code name of an API error, else the exception class name. Never the message."""
    code = getattr(exc, "code", None)
    if callable(code):
        try:
            c = code()
            return getattr(c, "name", None) or type(exc).__name__
        except Exception:  # noqa: BLE001
            pass
    return type(exc).__name__


# --------------------------------------------------------------------------- normalization

def split_curies(condition: str) -> list[str]:
    """`UBERON:0000178_CL:0000084` -> ['UBERON:0000178', 'CL:0000084'] (the client joins by _)."""
    parts = str(condition).split("_")
    bad = [x for x in parts if not CURIE_RE.match(x)]
    if bad:
        raise ValueError(f"condition {condition!r} does not split into CURIEs: {bad}")
    return parts


def channel_of(name: str, condition: str) -> str:
    prefix = f"{condition}_"
    if not str(name).startswith(prefix):
        raise ValueError(f"track name {name!r} does not start with {prefix!r}")
    return str(name)[len(prefix):]


def _target(output_type: str, row) -> str:
    if output_type == "CHIP_HISTONE":
        return str(row["histone_mark"])
    return ""


def normalize_prediction_channels(per_type: dict[str, pd.DataFrame], condition: str) -> pd.DataFrame:
    """Channels of a prediction's track metadata, one row per array column, in column order."""
    rows = []
    for ot in OUTPUT_TYPES:
        md = per_type.get(ot)
        if md is None:
            continue
        for idx, (_, r) in enumerate(md.reset_index(drop=True).iterrows()):
            rows.append({"output_type": ot, "channel_index": idx,
                         "channel": channel_of(r["name"], condition),
                         "target": _target(ot, r), "endedness": str(r["endedness"]),
                         "imputation_mode": str(r.get("imputation_mode", ""))})
    return pd.DataFrame(rows, columns=CHANNEL_COLUMNS + ["imputation_mode"])


def normalize(observed: dict[str, pd.DataFrame], supported: pd.DataFrame,
              channels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Observed metadata + supported conditions + prediction channels -> interface G tables.

    `observed` maps output type (CHIP_HISTONE, DNASE, ATAC) to the client's observed track
    metadata frame (columns name, biosample_curie, biosample_name, endedness, data_source,
    imputation_mode, histone_mark for ChIP). `supported` is `supported_conditions()`.
    """
    chan = channels[CHANNEL_COLUMNS].copy()
    chan["channel_index"] = chan["channel_index"].astype(int)
    if chan.duplicated(["output_type", "channel"]).any():
        raise ValueError("duplicate (output_type, channel) in the channel listing")
    index = {(r.output_type, r.channel): int(r.channel_index) for r in chan.itertuples()}

    rows = []
    for ot in OUTPUT_TYPES:
        df = observed.get(ot)
        if df is None or len(df) == 0:
            continue
        modes = set(df["imputation_mode"].astype(str)) if "imputation_mode" in df else {"observed"}
        if modes != {"observed"}:
            raise ValueError(f"{ot}: observed metadata holds imputation modes {sorted(modes)}")
        for _, r in df.iterrows():
            cond = str(r["biosample_curie"])
            split_curies(cond)
            ch = channel_of(r["name"], cond)
            if (ot, ch) not in index:
                raise ValueError(f"observed channel {ot} {ch!r} is not in the channel listing")
            rows.append({"output_type": ot, "condition": cond, "biosample_curie": cond,
                         "biosample_name": str(r["biosample_name"]), "target": _target(ot, r),
                         "endedness": str(r["endedness"]), "channel": ch,
                         "channel_index": index[(ot, ch)],
                         "data_source": str(r["data_source"])})
    obs = pd.DataFrame(rows, columns=OBSERVED_COLUMNS[:-1])
    keys = ["output_type", "condition", "channel"]
    counts = obs.groupby(keys, sort=False).size().rename("n_observed_tracks")
    obs = obs.drop_duplicates(keys).merge(counts.reset_index(), on=keys, how="left")
    obs = obs[OBSERVED_COLUMNS].sort_values(["output_type", "condition", "channel_index"],
                                            kind="stable").reset_index(drop=True)

    sup = supported.copy()
    if sup["condition"].duplicated().any():
        raise ValueError("duplicate condition in supported_conditions()")
    missing = sorted(set(obs["condition"]) - set(sup["condition"].astype(str)))
    if missing:
        raise ValueError(f"observed conditions missing from supported_conditions(): {missing[:5]}")
    bulk = obs[obs["data_source"] == ENCODE_DATA_SOURCE]
    n_bulk = bulk.groupby("condition")["n_observed_tracks"].sum()
    acc = set(obs.loc[obs["output_type"].isin(["DNASE", "ATAC"]), "condition"])
    cond = pd.DataFrame({
        "condition": sup["condition"].astype(str),
        "biosample_curies": [";".join(split_curies(c)) for c in sup["condition"].astype(str)],
        "biosample_name": sup["biosample_name"].astype(str),
    })
    cond["n_observed_encode_bulk"] = cond["condition"].map(n_bulk).fillna(0).astype(int)
    cond["has_observed_accessibility"] = cond["condition"].isin(acc).astype(int)
    # Second route to the same fact: the per-condition imputation modes of supported_conditions().
    for col in ("dnase", "atac"):
        if col in sup:
            via_modes = set(sup.loc[sup[col].astype(str) == "observed", "condition"].astype(str))
            via_obs = set(obs.loc[obs["output_type"] == col.upper(), "condition"])
            if via_modes != via_obs:
                raise ValueError(f"{col}: observed conditions disagree between "
                                 "observed_metadata() and supported_conditions()")
    cond = cond.sort_values("condition", kind="stable").reset_index(drop=True)
    return obs, chan.sort_values(["output_type", "channel_index"]).reset_index(drop=True), cond


def schema_markdown(obs: pd.DataFrame, chan: pd.DataFrame, cond: pd.DataFrame,
                    version: str) -> str:
    sources = sorted(set(obs["data_source"]))
    n_by_type = obs.groupby("output_type").size().to_dict()
    nch = chan.groupby("output_type").size().to_dict()
    return f"""# AlphaGenome 2 metadata tables: schema

Written by `tools/onion_data/ag2_meta.py metadata` (chunk AG0), client `alphagenome` {version},
organism human. The sources are two metadata calls, `observed_metadata(HOMO_SAPIENS)` and
`supported_conditions(HOMO_SAPIENS)`, and the track metadata of one prediction call
(`prediction_channels.tsv`, written by `ag2_meta.py smoke`). AG2 outputs are evaluation-only.

## The ENCODE data source

The `data_source` value that means ENCODE is `{ENCODE_DATA_SOURCE}`. The values seen in
`ag2_observed.tsv` are: {", ".join(f"`{s}`" for s in sources)}. An "ENCODE bulk track" of a
condition (Amendment 5 section B.1) is a row of `ag2_observed.tsv` with `data_source` equal to
`{ENCODE_DATA_SOURCE}` and `output_type` equal to `CHIP_HISTONE`, `DNASE` or `ATAC`.

## `ag2_observed.tsv` (one row per observed condition and channel)

Rows: {len(obs)} ({", ".join(f"{k} {v}" for k, v in sorted(n_by_type.items()))}).

| column | API source |
|---|---|
| `output_type` | the `observed_metadata()` field that held the row (`chip_histone`, `dnase`, `atac`), upper case |
| `condition` | `biosample_curie`; it equals the `condition` string of `supported_conditions()` and is the string a prediction call takes |
| `biosample_curie` | `biosample_curie` (CURIEs of a composite condition are joined by `_`) |
| `biosample_name` | `biosample_name` |
| `target` | `histone_mark` for `CHIP_HISTONE`; empty for `DNASE` and `ATAC` |
| `endedness` | `endedness` (`single`, `paired`) |
| `channel` | `name` without its leading `<biosample_curie>_` |
| `channel_index` | the column of `channel` in the array of a prediction call (`prediction_channels.tsv`) |
| `data_source` | `data_source` |
| `n_observed_tracks` | the count of `observed_metadata()` rows with this condition and channel (AG2 pools replicates, so it is 1) |

API columns not carried: `strand` (always `.`), `imputation_mode` (always `observed` here),
`biosample_type`, `Assay title`, `genetically_modified`, `data_pipeline`.

## `ag2_channels.tsv` (every channel of the three output types)

Rows: {len(chan)} ({", ".join(f"{k} {v}" for k, v in sorted(nch.items()))}). Source: the track
metadata of one prediction call, which lists every channel, observed or imputed, in array order.

| column | API source |
|---|---|
| `output_type` | the requested `OutputType` |
| `channel_index` | the row position in the prediction's track metadata, i.e. the column in its `values` array |
| `channel` | `name` without its leading `<condition>_` |
| `target` | `histone_mark` for `CHIP_HISTONE`; empty otherwise |
| `endedness` | `endedness` |

## `ag2_conditions.tsv` (every human reference condition the API supports)

Rows: {len(cond)}.

| column | API source |
|---|---|
| `condition` | `supported_conditions()` column `condition` |
| `biosample_curies` | `condition` split at `_`, joined by `;` |
| `biosample_name` | `supported_conditions()` column `biosample_name` |
| `n_observed_encode_bulk` | the sum of `n_observed_tracks` over this condition's rows of `ag2_observed.tsv` with `data_source` `{ENCODE_DATA_SOURCE}` |
| `has_observed_accessibility` | 1 when `ag2_observed.tsv` has a `DNASE` or `ATAC` row for the condition, else 0 |

The tool checks `has_observed_accessibility` by a second route: the conditions whose `dnase` or
`atac` column of `supported_conditions()` is `observed` equal the conditions with a `DNASE` or
`ATAC` row in `ag2_observed.tsv`.
"""


def write_tables(out: Path, obs, chan, cond, version: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    obs.to_csv(out / "ag2_observed.tsv", sep="\t", index=False)
    chan.to_csv(out / "ag2_channels.tsv", sep="\t", index=False)
    cond.to_csv(out / "ag2_conditions.tsv", sep="\t", index=False)
    (out / "META_SCHEMA.md").write_text(schema_markdown(obs, chan, cond, version))


def run_metadata(client, organism, out: Path, channels_tsv: Path, version: str):
    m = client.observed_metadata(organism)
    observed = {ot: getattr(m, ATTR[ot]) for ot in OUTPUT_TYPES}
    supported = client.supported_conditions(organism)
    channels = pd.read_csv(channels_tsv, sep="\t", keep_default_na=False)
    obs, chan, cond = normalize(observed, supported, channels)
    write_tables(out, obs, chan, cond, version)
    return obs, chan, cond


# --------------------------------------------------------------------------- smoke

def _metadata_frame(td):
    return td.metadata if td is not None else None


def run_smoke(client, api, out: Path, condition: str = SMOKE_CONDITION,
              max_calls: int = 6, chrom: str = "chr19") -> dict:
    """At most `max_calls` (<= 10) prediction calls on one condition; writes API_SMOKE.md."""
    if not 1 <= max_calls <= MAX_SMOKE_CALLS:
        raise SystemExit(f"ag2_meta: --max-calls must be 1..{MAX_SMOKE_CALLS}")
    out.mkdir(parents=True, exist_ok=True)
    m = client.observed_metadata(api.Organism.HOMO_SAPIENS)
    shapes_now = {ATTR[ot]: tuple(getattr(m, ATTR[ot]).shape) for ot in OUTPUT_TYPES}
    calls: list[dict] = []
    starts = iter(SMOKE_STARTS)

    def call(types: tuple[str, ...]):
        if len(calls) >= max_calls:
            raise RuntimeError("call budget spent")
        start = next(starts)
        iv = api.Interval(chrom, start, start + SMOKE_WINDOW)
        t0 = time.monotonic()
        rec = {"types": "+".join(types), "interval": f"{chrom}:{start}-{start + SMOKE_WINDOW}"}
        try:
            res = client.predict_interval(
                iv, organism=api.Organism.HOMO_SAPIENS,
                requested_outputs=[getattr(api.OutputType, t) for t in types],
                condition=condition)
            rec.update(ok=True, seconds=time.monotonic() - t0, error="")
        except Exception as exc:  # noqa: BLE001
            res = None
            rec.update(ok=False, seconds=time.monotonic() - t0, error=error_name(exc))
        calls.append(rec)
        return res

    combined = call(OUTPUT_TYPES)
    together_ok = combined is not None and all(
        getattr(combined, ATTR[t], None) is not None for t in OUTPUT_TYPES)
    per_type = {}
    if together_ok:
        per_type = {t: getattr(combined, ATTR[t]) for t in OUTPUT_TYPES}
    else:
        for t in OUTPUT_TYPES:
            if len(calls) >= max_calls:
                break
            r = call((t,))
            if r is not None and getattr(r, ATTR[t], None) is not None:
                per_type[t] = getattr(r, ATTR[t])
    # Timing: repeat the combined request (or ChIP only when combining fails) on new windows.
    while len(calls) < min(max_calls, 6):
        call(OUTPUT_TYPES if together_ok else ("CHIP_HISTONE",))

    chan = normalize_prediction_channels({t: _metadata_frame(td) for t, td in per_type.items()},
                                         condition)
    chan.to_csv(out / "prediction_channels.tsv", sep="\t", index=False)

    facts = {"version": api.version, "condition": condition, "calls": calls,
             "together": together_ok, "shapes_now": shapes_now}
    chip = per_type.get("CHIP_HISTONE")
    if chip is not None:
        v = np.asarray(chip.values, dtype=np.float64)
        med = np.median(v, axis=0)
        facts["chip"] = {
            "shape": tuple(v.shape), "resolution": int(getattr(chip, "resolution", 0)),
            "median_of_channel_medians": float(np.median(med)),
            "p10_p90_channel_medians": (float(np.percentile(med, 10)),
                                        float(np.percentile(med, 90))),
            "median_all": float(np.median(v)), "max_all": float(v.max())}
    for t in ("DNASE", "ATAC"):
        td = per_type.get(t)
        if td is not None:
            v = np.asarray(td.values, dtype=np.float64)
            facts[t.lower()] = {"shape": tuple(v.shape),
                                "resolution": int(getattr(td, "resolution", 0)),
                                "median_window_sum": float(np.median(v.sum(axis=0)))}
    md_cols = {t: list(_metadata_frame(td).columns) for t, td in per_type.items()}
    facts["imputation_mode_in"] = {t: ("imputation_mode" in cols) for t, cols in md_cols.items()}
    facts["imputation_modes"] = chan.groupby(["output_type", "imputation_mode"]).size().to_dict()
    (out / "API_SMOKE.md").write_text(smoke_markdown(facts))
    return facts


def chip_verdict(median_of_medians: float) -> str:
    """Per-bp fold change is about 1 in background; a 128 bp sum of it is about 128."""
    if median_of_medians < 8:
        return "mean"
    if median_of_medians > 32:
        return "sum"
    return "undecided"


def smoke_markdown(f: dict) -> str:
    ok = [c for c in f["calls"] if c["ok"]]
    secs = [c["seconds"] for c in ok]
    lines = ["# AlphaGenome 2 API smoke (chunk AG0)", "",
             f"Client `alphagenome` {f['version']}; condition `{f['condition']}`; "
             f"{len(f['calls'])} prediction calls on 1,048,576 bp windows; written by "
             "`tools/onion_data/ag2_meta.py smoke`. No prediction array was saved. AG2 outputs "
             "are evaluation-only.", "",
             "## 1. Several output types in one call", "",
             f"one call, several output types: {'yes' if f['together'] else 'no'}", ""]
    c0 = f["calls"][0]
    lines.append(f"The first call requested `CHIP_HISTONE`, `DNASE` and `ATAC` together; it "
                 f"{'returned all three' if f['together'] else 'failed or lacked an output'}"
                 f"{'' if c0['ok'] else ' (status ' + c0['error'] + ')'}.")
    lines += ["", "## 2. ChIP-seq value per 128 bp bin: sum or mean", ""]
    chip = f.get("chip")
    if chip:
        v = chip_verdict(chip["median_of_channel_medians"])
        lines.append(
            f"Verdict: **{v}** of the per-bp fold change. Evidence: ChIP array shape "
            f"{chip['shape']}, resolution {chip['resolution']} bp. The median over channels of "
            f"each channel's median bin value is {chip['median_of_channel_medians']:.3f} (10th "
            f"to 90th percentile of channel medians {chip['p10_p90_channel_medians'][0]:.3f} to "
            f"{chip['p10_p90_channel_medians'][1]:.3f}; largest bin {chip['max_all']:.1f}). A "
            "fold change over control is about 1 in background, so a mean per bin gives a "
            "median near 1 and a sum over 128 bp gives a median near 128. The rule is: median "
            "below 8 means mean, above 32 means sum, else undecided.")
    else:
        lines.append("Not measured: no ChIP-seq output was returned.")
    for t in ("dnase", "atac"):
        if t in f:
            exp = 1e8 * SMOKE_WINDOW / 3.1e9
            lines.append("")
            lines.append(
                f"For scale, {t.upper()} array shape {f[t]['shape']}, resolution "
                f"{f[t]['resolution']} bp; the median over channels of the 1 Mb window sum is "
                f"{f[t]['median_window_sum']:.0f}, against {exp:.0f} for a uniform track that "
                "sums to 1e8 over a 3.1e9 bp genome.")
    lines += ["", "## 3. Where the `imputation_mode` label is read", ""]
    where = ", ".join(f"{t}: {'yes' if v else 'no'}" for t, v in f["imputation_mode_in"].items())
    lines.append("The label is the column `imputation_mode` of the track metadata of each "
                 "returned output, i.e. `output.<type>.metadata['imputation_mode']`, one value "
                 f"per channel. Present per output type: {where}. Counts per channel for this "
                 "condition: " + ", ".join(f"{k[0]} {k[1]} {n}"
                                           for k, n in sorted(f["imputation_modes"].items()))
                 + ". `supported_conditions()` also gives one mode per condition and output "
                 "type (a column per output type).")
    lines += ["", "## 4. Seconds per call", ""]
    if secs:
        lines.append(f"Seconds per successful call: {', '.join(f'{s:.1f}' for s in secs)} "
                     f"(median {float(np.median(secs)):.1f}; {len(secs)} of {len(f['calls'])} "
                     "calls succeeded; wall time from a Nibi login node, one call at a time).")
    else:
        lines.append("No call succeeded.")
    lines += ["", "## Calls", "", "| # | outputs | interval | ok | seconds | error |",
              "|---|---|---|---|---|---|"]
    for i, c in enumerate(f["calls"], 1):
        lines.append(f"| {i} | {c['types']} | {c['interval']} | {'yes' if c['ok'] else 'no'} | "
                     f"{c['seconds']:.1f} | {c['error'] or '-'} |")
    lines += ["", "## Observed metadata shapes", "",
              "Seen on 2026-11-12: " + ", ".join(f"{k} {v}" for k, v in
                                                SHAPES_SEEN_2026_11_12.items()) + ".",
              "Seen by this run: " + ", ".join(f"{k} {v}" for k, v in f["shapes_now"].items())
              + ".", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--condition", default=SMOKE_CONDITION)
    s.add_argument("--max-calls", type=int, default=6)
    m = sub.add_parser("metadata")
    m.add_argument("--out", required=True, type=Path)
    m.add_argument("--channels", type=Path, default=None,
                   help="prediction_channels.tsv (default <out>/prediction_channels.tsv)")
    a = ap.parse_args(argv)
    api = load_api()
    client = make_client(api)
    if a.cmd == "smoke":
        f = run_smoke(client, api, a.out, a.condition, a.max_calls)
        print(f"smoke: {len(f['calls'])} calls; together={f['together']}; wrote {a.out}")
    else:
        ch = a.channels or a.out / "prediction_channels.tsv"
        if not ch.exists():
            raise SystemExit(f"ag2_meta: {ch} missing; run `ag2_meta.py smoke --out {a.out}` first")
        obs, chan, cond = run_metadata(client, api.Organism.HOMO_SAPIENS, a.out, ch, api.version)
        print("metadata: observed " + ", ".join(
            f"{k} {v}" for k, v in obs.groupby("output_type").size().items())
              + f"; channels {len(chan)}; conditions {len(cond)}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
