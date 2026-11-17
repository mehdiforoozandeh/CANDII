"""Close-out item A12: arm configs that drop covariate fields, from ANY base config.

    python tools/minicandi/covariate_field_arms.py lofo   --base <cfg.json> [--base ...] --out <dir>
    python tools/minicandi/covariate_field_arms.py groups --base <cfg.json> [--base ...] --out <dir>
    python tools/minicandi/covariate_field_arms.py curve  --base <cfg.json> [--base ...] --out <dir> \\
        --order <field>,<field>,...        # least useful first (covariate_ranking.py writes it)

Each base config (one per model; its "rung" key picks the folder) gives `<out>/rung<r>/<arm>.json`
= the base config plus one key, `covariate_drop_fields` (minicandi.covariates.DROP_FIELDS_KEY), and
one line per arm in `<out>/arms.tsv` (arm, rung, kind, fields, base path, base sha256). The folder
is the shape `configs/ablations/submit_all.sh` reads (ABLATIONS=<folder name>).

- lofo: one arm per field, `lofo_<field>`. Fields: the 11 META_COLUMNS, plus the 7 QC_COLUMNS when
  the base has qc_on_input; `--skip` removes fields (e.g. a QC column empty in every row).
- groups: the grouped-removal control for correlated fields (GROUPS), `group_<name>`.
- curve: the rank-guided backward elimination, `curve_k<NN>` drops the first NN fields of
  `--order`, for NN = 2 .. len(order) (NN = 1 is the LOFO arm of the first field).

A base that already drops a field is refused. A two_step base is accepted: the switch drops the
field in both legs (`drop_field_indices`).
Writes only under --out; an existing arm file with other content is refused (never overwritten).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from minicandi.contract import META_COLUMNS, QC_COLUMNS  # noqa: E402
from minicandi.covariates import DROP_FIELDS_KEY, drop_field_indices  # noqa: E402

#: the grouped-removal control (A12 specification, "Controls and traps"). Each group joins fields
#: whose information overlaps in the eic_v1_2 training rows (conditional entropy H(a | b) / H(a),
#: the A12 item file, section "Fields"), so that a leave-one-out arm of one member cannot show
#: the group's value while the other member still carries it.
GROUPS = {
    "pipeline": ("pipeline_id", "pipeline_version_id"),       # identical information (0.00)
    "lab_platform": ("lab_id", "platform_id"),                # the brief's pair (0.57 / 0.69)
    "donor_lab": ("donor_id", "lab_id"),                      # donor fixes lab (0.15)
    "view_knobs": ("run_type_pe", "read_length", "dedup_on",  # the processing knobs that the
                   "mapq_threshold"),                         # counterfactual views vary (se, crop,
                                                              # nodedup, mapq; depth is the offset)
    "categorical": ("pipeline_id", "pipeline_version_id", "lab_id", "platform_id",
                    "antibody_id", "donor_id"),               # = the numeric-only embedding arm
    "all_but_depth": tuple(c for c in META_COLUMNS if c != "log2_depth"),
}
#: the groups whose arm the "safe to remove" rule reads (covariate_ranking.classify): fields whose
#: information overlaps. `categorical` and `all_but_depth` are reference arms, read beside the
#: curve: a useful `all_but_depth` says only that not every field can go at once.
CORRELATED_GROUPS = ("pipeline", "lab_platform", "donor_lab", "view_knobs")
KINDS = ("lofo", "groups", "curve")


def fields_of(base: dict, skip=()) -> list:
    cols = list(META_COLUMNS) + (list(QC_COLUMNS) if base.get("qc_on_input") else [])
    bad = [s for s in skip if s not in cols]
    if bad:
        raise ValueError(f"--skip {bad}: not a field of this base ({cols})")
    return [c for c in cols if c not in skip]


def arms_for(kind: str, base: dict, skip=(), order=None) -> list:
    """[(arm name, [fields])] of one kind for one base config."""
    if base.get(DROP_FIELDS_KEY):
        raise ValueError(f"the base already has {DROP_FIELDS_KEY} = {base[DROP_FIELDS_KEY]}")
    fields = fields_of(base, skip)
    if kind == "lofo":
        out = [(f"lofo_{f}", [f]) for f in fields]
    elif kind == "groups":
        out = [(f"group_{g}", list(fs)) for g, fs in GROUPS.items()
               if all(f in fields for f in fs)]
        if base.get("qc_on_input"):
            qc = [c for c in QC_COLUMNS if c in fields]
            if qc:
                out.append(("group_qc", qc))
    elif kind == "curve":
        if not order:
            raise ValueError("curve needs --order (least useful field first)")
        if len(set(order)) != len(order) or any(f not in fields for f in order):
            raise ValueError(f"--order {order}: distinct fields of this base ({fields})")
        out = [(f"curve_k{k:02d}", list(order[:k])) for k in range(2, len(order) + 1)]
    else:
        raise ValueError(f"kind {kind!r} not in {KINDS}")
    for _, fs in out:                       # the switch's own validation (QC, names)
        drop_field_indices({**base, DROP_FIELDS_KEY: fs})
    return out


def write_arms(kind: str, bases, out: Path, skip=(), order=None) -> list:
    """Write every arm config of `kind` for each base path in `bases`; return the arms.tsv rows."""
    out = Path(out)
    rows, seen = [], set()
    for bp in bases:
        bp = Path(bp)
        raw = bp.read_bytes()
        base = json.loads(raw)
        r = base.get("rung")
        if r in seen:
            raise ValueError(f"two base configs for rung {r}")
        seen.add(r)
        for name, fs in arms_for(kind, base, skip, order):
            cfg = {**base, DROP_FIELDS_KEY: fs}
            p = out / f"rung{r}" / f"{name}.json"
            text = json.dumps(cfg, indent=1) + "\n"
            if p.exists() and p.read_text(encoding="utf-8") != text:
                raise FileExistsError(f"{p} exists with other content; use a new --out")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            rows.append([name, str(r), kind, ",".join(fs), str(bp), hashlib.sha256(raw).hexdigest()])
    tsv = out / "arms.tsv"
    old = tsv.read_text(encoding="utf-8").splitlines()[1:] if tsv.is_file() else []
    lines = sorted(set(old) | {"\t".join(x) for x in rows})
    tsv.write_text("arm\trung\tkind\tfields\tbase\tbase_sha256\n" + "".join(l + "\n" for l in lines),
                   encoding="utf-8")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=KINDS)
    ap.add_argument("--base", action="append", required=True, help="base config (one per model)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", default="", help="comma-separated fields to leave out")
    ap.add_argument("--order", default="", help="curve: comma-separated fields, least useful first")
    a = ap.parse_args(argv)
    skip = [s for s in a.skip.split(",") if s]
    order = [s for s in a.order.split(",") if s] or None
    rows = write_arms(a.kind, a.base, Path(a.out), skip, order)
    for x in rows:
        print("\t".join(x[:4]))
    print(f"{len(rows)} arm configs ({a.kind}) under {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
