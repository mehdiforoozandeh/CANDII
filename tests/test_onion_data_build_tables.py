"""t152 chunk D6t: the tables builder on a tiny tree of real `align_views.py` measurement dirs (CPU,
seconds). The python splitter stands in for the pipeline's spr split; the depth arms are thinned from
the base halves, and one FASTQ-level arm (`se`) is split on its own, as a tagAlign-route arm would be.

Tree: a validation H3K27ac track with 2 isogenic replicates and its control, two training tracks, a
ledger `match = 0` experiment, and a blind testbed track (2 arms) given as `testbed_rows.tsv`.
"""
from __future__ import annotations

import csv
import gzip
import json
import shutil
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import align_views as AV  # noqa: E402
from onion_data import build_list as BL  # noqa: E402
from onion_data import build_tables as BT  # noqa: E402
from onion_data import eic_accessions as EA  # noqa: E402
from onion_data import import_testbed as IT  # noqa: E402
from minicandi.dataset import MEASUREMENT_COLUMNS, PAIR_COLUMNS, read_tsv  # noqa: E402
from minicandi.covariates import load_covariates_json  # noqa: E402

SIZES = {"chr1": 409600, "chr21": 409600}          # 16384 bins each: one 2^14 block per chrom
VAL, TRN, TRN2, BAD, CTL, BLD = ("ENCSRVAL001", "ENCSRTRN002", "ENCSRTRN003", "ENCSRBAD004",
                                 "ENCSRCTL005", "ENCSRBLD006")
EXPS = {  # experiment: (assay, cell, track, role, sources, arms, layer, match, extra)
    VAL: ("H3K27ac", "C01", "C01M01", "val", 2, "base,depth0.50,depth0.25", "uniform", "1"),
    TRN: ("H3K27ac", "C02", "C02M01", "train", 1, "base,depth0.50,depth0.25", "uniform", "1"),
    TRN2: ("H3K4me3", "C01", "C01M02", "train", 1, "base", "uniform", "1"),
    BAD: ("H3K4me3", "C02", "C02M02", "train", 1, "base", "uniform", "0"),
    CTL: ("chipseq-control", "C01", "C01CTL", "val", 1, "base", "uniform", "1"),
    BLD: ("H3K27ac", "C03", "C03M01", "blind", 1, "base,depth0.50", "testbed", "1"),
}
WITHHELD = [{"source_knob": "depth", "source_value": "0.25", "target_knob": "depth",
             "target_value": "0.50", "assay_class": "narrow_histone"}]


def _ta(path: Path, n: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for i in range(n):
            c = "chr1" if rng.random() < 0.5 else "chr21"
            s = int(rng.integers(0, SIZES[c] - 50))
            f.write(f"{c}\t{s}\t{s + 36}\tr{seed}_{i}\t1000\t{'+-'[i % 2]}\n")
    return path


def _av_build(tmp, exp, spec, sources, *, source_arm=None):
    assay, cell, track, role, _, arms, layer, _ = spec
    argv = ["build", "--experiment", exp, "--assay", assay, "--cell", cell, "--eic-track", track,
            "--role", role, "--layer", layer, "--arms", arms, "--splitter", "python",
            "--work", str(tmp / "work"), "--out", str(tmp / "measurements"),
            "--chrom-sizes", str(tmp / "chrom_sizes.json"), "--tmp", str(tmp / "tmp"),
            "--read-length", "36", "--meta", "lab=LabA", "--meta", "platform=Illumina"]
    for s in sources:
        argv += ["--source", str(s)]
    if source_arm is None:
        assert AV.main(argv) == 0
        return
    orig = AV.build

    def patched(cfg):
        cfg["source_arm"], cfg["arms"] = source_arm, ["base"]
        return orig(cfg)
    with mock.patch.object(AV, "build", patched):
        assert AV.main(argv) == 0


def _write_tsv(path, columns, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def _tok(mids, val=False):
    return ";".join(BL.token(m, BL.BASE_PRODUCTS + (BL.SIGNAL_PRODUCTS if val and m.endswith(
        "pooled.uniform.base") else ())) for m in mids)


def make_tree(tmp: Path) -> dict:
    tmp = Path(tmp)
    (tmp / "tmp").mkdir(parents=True, exist_ok=True)
    (tmp / "chrom_sizes.json").write_text(json.dumps({"chrom_sizes": SIZES}))
    seed = 0
    for exp, spec in EXPS.items():
        srcs = []
        for i in range(spec[4]):
            seed += 1
            srcs.append(_ta(tmp / "src" / f"{exp}_rep{i + 1}.tagAlign.gz", 3000, seed))
        _av_build(tmp, exp, spec, srcs)
    # a FASTQ-level arm built from its own read set, split on its own (shares reads with base)
    _av_build(tmp, TRN, EXPS[TRN], [_ta(tmp / "src" / f"{TRN}_se.tagAlign.gz", 2500, 99)],
              source_arm="se")
    # QC of the training base run (qc.json) and of the validation run's rep1 (xcor file only)
    run = tmp / "work" / TRN / "base"
    run.mkdir(parents=True, exist_ok=True)
    (run / "qc.json").write_text(json.dumps({
        "lib_complexity": {"lib_complexity": {"rep1": {"NRF": 0.91, "PBC1": 0.9, "PBC2": 12.5}}},
        "align_enrich": {"xcor_score": {"rep1": {"NSC": 1.5, "RSC": 1.2,
                                                 "estimated_fragment_len": 180}}}}))
    run = tmp / "work" / VAL / "base"
    run.mkdir(parents=True, exist_ok=True)
    (run / "rep1.xcor.qc").write_text("f.tagAlign.gz\t1000\t155\t0.3\t40\t0.27\t1500\t0.16\t"
                                      "1.93\t1.40\t1\n")

    led = []
    for exp, (assay, cell, track, role, *_rest) in EXPS.items():
        match = _rest[-1]
        r = {c: "" for c in EA.LEDGER_COLUMNS}
        r.update({"eic_track": track, "role": role, "cell": cell, "assay": assay,
                  "assay_class": AV_class(assay), "experiment": exp, "match": match,
                  "published_experiment": exp if match == "1" and assay != "chipseq-control" else "",
                  "replication_type": "isogenic" if exp == VAL else "unreplicated",
                  "encode_possible_controls": CTL if exp == VAL else ""})
        led.append(r)
    _write_tsv(tmp / "ledger.tsv", EA.LEDGER_COLUMNS, led)
    with open(tmp / "published.tsv", "w") as f:
        f.write("# synthetic published list\n")
        f.write("data_type\tcell_type_id\tmark_id\tcell_type\tassay\tfilename\texperiment\n")
        dt = {"train": "training_data", "val": "validation_data", "blind": "blind_data"}
        for exp, (assay, cell, track, role, *_r) in EXPS.items():
            if assay != "chipseq-control" and exp != BAD:
                f.write(f"{dt[role]}\t{cell}\t{track[3:]}\tcell_{cell}\t{assay}\t{track}\t{exp}\n")

    def mids(exp, arms, units=("pooled",)):
        return [f"{exp}.{u}.uniform.{a}" for u in units for a in arms]
    bl = [
        (1, CTL, "base", mids(CTL, ["base"])),
        (1, VAL, "base", mids(VAL, ["base", "depth0.50", "depth0.25"]) + mids(VAL, ["base"], ("br1", "br2"))),
        (2, TRN, "base", mids(TRN, ["base", "depth0.50", "depth0.25"])),
        (2, TRN2, "base", mids(TRN2, ["base", "depth0.50"])),          # depth0.50 never built
        (2, BAD, "base", mids(BAD, ["base"])),
        (3, TRN, "se", mids(TRN, ["se"])),
        (4, BLD, "base", mids(BLD, ["base"])),                         # priority 4: not selected
    ]
    rows = []
    for i, (p, exp, arm, ms) in enumerate(bl, 1):
        assay, cell, track, role, *_ = EXPS[exp]
        rows.append({"row": str(i), "priority": str(p), "experiment": exp, "eic_track": track,
                     "role": role, "assay": assay, "assay_class": AV_class(assay), "unit": "pooled",
                     "arm": arm, "pipeline": "chip-seq-pipeline2 v2.2.2",
                     "expected_measurements": _tok(ms, val=(role == "val"))})
    _write_tsv(tmp / "build_list.tsv", BL.BUILD_LIST_COLUMNS, rows)
    _write_tsv(tmp / "withheld.tsv", BL.WITHHELD_COLUMNS, WITHHELD)
    tb = []
    for arm in ("base", "depth0.50"):
        d = tmp / "measurements" / f"{BLD}.pooled.testbed.{arm}"
        r = json.loads((d / "meta.json").read_text())
        r["use"] = IT.USE_BLIND
        tb.append(r)
    _write_tsv(tmp / "testbed_rows.tsv", IT.row_columns(), tb)
    return {"root": tmp, "measurements": tmp / "measurements", "work": tmp / "work",
            "ledger": tmp / "ledger.tsv", "published": tmp / "published.tsv",
            "build_list": tmp / "build_list.tsv", "withheld": tmp / "withheld.tsv",
            "testbed_rows": tmp / "testbed_rows.tsv"}


def AV_class(assay):
    from minicandi import contract as C
    return C.assay_class(assay)


def build_argv(t: dict, out, priorities="1,2,3", *extra) -> list:
    return ["build", "--build-list", str(t["build_list"]), "--priorities", priorities,
            "--measurements", str(t["measurements"]), "--work", str(t["work"]),
            "--testbed-rows", str(t["testbed_rows"]), "--withheld", str(t["withheld"]),
            "--ledger", str(t["ledger"]), "--out", str(out), *extra]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    t = make_tree(tmp_path_factory.mktemp("tree"))
    out = t["root"] / "tables" / "v1"
    assert BT.main(build_argv(t, out)) == 0
    return t, out


def _rows(out):
    return {r["measurement_id"]: r for r in read_tsv(out / "measurements.tsv", MEASUREMENT_COLUMNS)}


def _pairs(out, name="pairs.tsv"):
    return read_tsv(out / name, PAIR_COLUMNS)


def test_rows_only_from_verified_ledger_dirs(built):
    t, out = built
    rows = _rows(out)
    assert f"{BAD}.pooled.uniform.base" not in rows                  # ledger match = 0
    assert f"{BLD}.pooled.uniform.base" not in rows                  # priority 4 not selected
    assert f"{VAL}.br1.uniform.base" in rows and f"{TRN}.pooled.uniform.se" in rows
    miss = {m["measurement_id"]: m for m in read_tsv(out / "missing.tsv")}
    assert "not built" in miss[f"{TRN2}.pooled.uniform.depth0.50"]["reason"]
    assert "ledger" in miss[f"{BAD}.pooled.uniform.base"]["reason"]
    assert miss[f"{VAL}.pooled.uniform.base"]["products_missing"] == "signal+pr1_signal+pr2_signal"
    assert f"{VAL}.pooled.uniform.base" in rows                      # kept, signal listed
    for m, r in rows.items():                                         # rows are the meta.json rows
        meta = json.loads((Path(r["counts_path"][3:]).parent / "meta.json").read_text())
        assert {k: v for k, v in meta.items() if k not in BT.QC_FROM_JSON.values()} == \
            {k: v for k, v in r.items() if k not in BT.QC_FROM_JSON.values()}


def test_testbed_rows_kept_blind_and_flagged_evaluation_only(built):
    t, out = built
    rows = _rows(out)
    tb = [r for r in rows.values() if r["layer"] == "testbed"]
    assert len(tb) == 2 and all(r["in_eic"] == "blind" for r in tb)
    ev = {r["measurement_id"]: r["use"] for r in read_tsv(out / "evaluation_only.tsv")}
    assert set(ev) == {r["measurement_id"] for r in tb} and set(ev.values()) == {IT.USE_BLIND}
    shared = _pairs(out, "pairs_shared_reads.tsv")
    assert any(p["a"].startswith(BLD) and p["b"].startswith(BLD) for p in shared)
    assert not any(p["rung"] == "designed" and p["a"].startswith(BLD) for p in _pairs(out))


def test_pairs_rungs_split_withheld_and_split_signature(built):
    t, out = built
    P = _pairs(out)
    by = {(p["rung"], p["a"], p["b"]): p for p in P}
    assert ("pseudoreplicate", f"{VAL}.pooled.uniform.base", f"{VAL}.pooled.uniform.base") in by
    iso = by[("isogenic", f"{VAL}.br1.uniform.base", f"{VAL}.br2.uniform.base")]
    assert iso["split"] == "val" and iso["cell"] == "C01"
    oc = by[("other_cell_type", f"{VAL}.pooled.uniform.base", f"{TRN}.pooled.uniform.base")]
    assert oc["split"] == "val" and oc["cell"] == ""
    assert not any(p["rung"] == "other_cell_type" and "testbed" in p["a"] + p["b"] for p in P)
    nt = by[("designed", f"{VAL}.pooled.uniform.depth0.25", f"{VAL}.pooled.uniform.depth0.50")]
    assert nt["never_trained"] == "1" and nt["subset_pair"] == "1" and nt["split"] == "val"
    back = by[("designed", f"{VAL}.pooled.uniform.depth0.50", f"{VAL}.pooled.uniform.depth0.25")]
    assert back["never_trained"] == "0"
    # the same combination on a training track is never written
    assert ("designed", f"{TRN}.pooled.uniform.depth0.25", f"{TRN}.pooled.uniform.depth0.50") not in by
    assert ("designed", f"{TRN}.pooled.uniform.base", f"{TRN}.pooled.uniform.depth0.50") in by
    # se was split on its own: its pairs with base go to the shared-reads file only
    assert not any(p["rung"] == "designed" and (p["a"].endswith(".se") or p["b"].endswith(".se"))
                   for p in P)
    shared = {(p["a"], p["b"]) for p in _pairs(out, "pairs_shared_reads.tsv")}
    assert (f"{TRN}.pooled.uniform.base", f"{TRN}.pooled.uniform.se") in shared
    assert len({p["pair_id"] for p in P}) == len(P)


def test_split_signature_inherits_through_thinning(built):
    t, out = built
    rows = _rows(out)
    sig = BT.SplitSignature()
    b, d, se = (rows[f"{TRN}.pooled.uniform.{a}"] for a in ("base", "depth0.25", "se"))
    assert sig.one_split(b, d) and not sig.one_split(b, se)


def test_qc_from_run_directory(built):
    t, out = built
    rows = _rows(out)
    r = rows[f"{TRN}.pooled.uniform.base"]
    assert (r["nrf"], r["pbc1"], r["pbc2"], r["nsc"], r["rsc"], r["frag_len"]) == \
        ("0.91", "0.9", "12.5", "1.5", "1.2", "180")
    assert rows[f"{VAL}.br1.uniform.base"]["frag_len"] == "155"
    assert rows[f"{VAL}.br1.uniform.base"]["nsc"] == "1.93"
    assert rows[f"{VAL}.pooled.uniform.base"]["nsc"] == ""          # 2-replicate pool: left empty
    assert rows[f"{TRN}.pooled.uniform.depth0.25"]["nsc"] == ""     # derived arm: left empty
    assert all(r["frip"] == "" for r in rows.values())


def test_covariates_and_withheld_copied(built):
    t, out = built
    spec = load_covariates_json(out / "covariates.json")
    assert spec["vocab"]["lab"] == ["LabA"]
    assert (out / "withheld.tsv").read_bytes() == Path(t["withheld"]).read_bytes()
    info = json.loads((out / "build_info.json").read_text())
    assert info["ledger"] == str(Path(t["ledger"]).resolve())


def test_out_must_be_empty_and_dry_run_writes_nothing(built, tmp_path, capsys):
    t, out = built
    assert BT.main(build_argv(t, out)) == 2                           # frozen tables: refused
    o2 = tmp_path / "o2"
    o2.mkdir()
    shutil.copy(t["withheld"], o2 / "withheld.tsv")                   # only withheld.tsv: allowed
    assert BT.main(build_argv(t, o2)) == 0
    o3 = tmp_path / "o3"
    o3.mkdir()
    (o3 / "withheld.tsv").write_text("source_knob\n")                 # a different withheld.tsv
    assert BT.main(build_argv(t, o3)) == 2
    capsys.readouterr()
    dry = tmp_path / "dry"
    assert BT.main(build_argv(t, dry, "1,2,3", "--dry-run")) == 0
    txt = capsys.readouterr().out
    assert not dry.exists()
    assert "measurement directories read: " in txt and "rows that would be written: " in txt
    n = int(txt.split("rows that would be written: ")[1].split()[0])
    assert n == len(_rows(out))


def test_superset_keeps_every_older_row(built, tmp_path):
    t, out = built
    v1 = tmp_path / "v1"
    assert BT.main(build_argv(t, v1, "1,2")) == 0
    v2 = tmp_path / "v2"
    assert BT.main(build_argv(t, v2, "1,2,3", "--superset-of", str(v1))) == 0
    old = set(BT.body_lines(v1 / "measurements.tsv"))
    assert old < set(BT.body_lines(v2 / "measurements.tsv"))
    assert set(BT.body_lines(v1 / "pairs.tsv")) <= set(BT.body_lines(v2 / "pairs.tsv"))
    # a changed older row refuses the build and writes nothing
    lines = (v1 / "measurements.tsv").read_text().splitlines(keepends=True)
    lines[1] = lines[1].replace("LabA", "LabZ")
    (v1 / "measurements.tsv").write_text("".join(lines))
    v3 = tmp_path / "v3"
    assert BT.main(build_argv(t, v3, "1,2,3", "--superset-of", str(v1))) == 1
    assert not v3.exists()


def test_failed_verify_goes_to_missing(tmp_path):
    t = make_tree(tmp_path)
    d = t["measurements"] / f"{TRN2}.pooled.uniform.base"
    meta = json.loads((d / "meta.json").read_text())
    meta["sha256"] = "0" * 64
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    out = tmp_path / "tables"
    assert BT.main(build_argv(t, out)) == 0
    assert f"{TRN2}.pooled.uniform.base" not in _rows(out)
    miss = {m["measurement_id"]: m["reason"] for m in read_tsv(out / "missing.tsv")}
    assert miss[f"{TRN2}.pooled.uniform.base"].startswith("verify failed: ")
    assert "sha256" in miss[f"{TRN2}.pooled.uniform.base"]
