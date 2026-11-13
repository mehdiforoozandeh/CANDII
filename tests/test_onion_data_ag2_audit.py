"""Tests for the AG2 leakage audit (onion Amendment 5, chunk AGl). Synthetic tables only; no AG2 client."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from onion_data import ag2_audit as A  # noqa: E402

LEDGER_COLS = ["eic_track", "role", "cell", "assay", "assay_class", "experiment", "run_type"]
SE, PE = "single-ended", "paired-ended"

# Conditions: two single-term conditions, two composites that both contain CL:3, one composite holding CL:4.
CONDITIONS = [
    {"condition": "CL:1", "biosample_curies": "CL:1", "biosample_name": "one", "n_observed_encode_bulk": "5",
     "has_observed_accessibility": "1"},
    {"condition": "CL:2", "biosample_curies": "CL:2", "biosample_name": "two", "n_observed_encode_bulk": "1",
     "has_observed_accessibility": "0"},
    {"condition": "CL:3_CL:9", "biosample_curies": "CL:3;CL:9", "biosample_name": "small",
     "n_observed_encode_bulk": "2", "has_observed_accessibility": "0"},
    {"condition": "CL:3_CL:8", "biosample_curies": "CL:3;CL:8", "biosample_name": "big",
     "n_observed_encode_bulk": "7", "has_observed_accessibility": "1"},
    {"condition": "CL:4", "biosample_curies": "CL:4", "biosample_name": "four", "n_observed_encode_bulk": "0",
     "has_observed_accessibility": "0"},
    {"condition": "CL:4_CL:7", "biosample_curies": "CL:4;CL:7", "biosample_name": "fourplus",
     "n_observed_encode_bulk": "3", "has_observed_accessibility": "0"},
    {"condition": "EFO:5", "biosample_curies": "EFO_5", "biosample_name": "Caco-2", "n_observed_encode_bulk": "1",
     "has_observed_accessibility": "0"},
    {"condition": "CL:10", "biosample_curies": "CL:10", "biosample_name": "ten", "n_observed_encode_bulk": "0",
     "has_observed_accessibility": "0"},
]
CHANNELS = [
    {"output_type": "CHIP_HISTONE", "channel_index": "0", "channel": "K27ac_se", "target": "H3K27ac",
     "endedness": "single"},
    {"output_type": "CHIP_HISTONE", "channel_index": "1", "channel": "K27ac_pe", "target": "H3K27ac",
     "endedness": "paired"},
    {"output_type": "CHIP_HISTONE", "channel_index": "2", "channel": "H2AZ_se", "target": "H2A.Z",
     "endedness": "single"},
    {"output_type": "DNASE", "channel_index": "0", "channel": "dnase", "target": "", "endedness": ""},
]


def obs(cond, ot, target, end, channel, source="encode"):
    return {"output_type": ot, "condition": cond, "biosample_curie": cond, "biosample_name": "",
            "target": target, "endedness": end, "channel": channel, "channel_index": "0", "data_source": source,
            "n_observed_tracks": "1"}


OBSERVED = [
    obs("CL:1", "CHIP_HISTONE", "H3K27AC", "single", "K27ac_se"),     # CL:1 K27ac single (case differs: fix 1)
    obs("CL:1", "DNASE", "", "", "dnase"),
    obs("CL:2", "CHIP_HISTONE", "H3K27ac", "paired", "K27ac_pe"),     # CL:2 K27ac paired only
    obs("CL:4_CL:7", "CHIP_HISTONE", "H3K27ac", "single", "K27ac_se"),  # sibling of CL:4
    obs("CL:3_CL:8", "CHIP_HISTONE", "H2A.Z", "single", "H2AZ_se"),
    obs("CL:4", "CHIP_HISTONE", "H2A.Z", "single", "H2AZ_se"),         # CL:4 has a trained assay
    obs("EFO:5", "CHIP_HISTONE", "H3K27ac", "single", "K27ac_se"),     # Caco-2 under another CURIE
]


def L(track, role, cell, assay, cls, exp, run_type=SE):
    return dict(zip(LEDGER_COLS, [track, role, cell, assay, cls, exp, run_type]))


LEDGER = [
    L("T01", "val", "one", "H3K27ac", "narrow_histone", "E1"),             # observed_matched
    L("T02", "val", "two", "H3K27ac", "narrow_histone", "E2"),             # observed_other_read_type
    L("T03", "blind", "four", "H3K27ac", "narrow_histone", "E4", PE),       # sibling_condition
    L("T04", "blind", "one", "H2AFZ", "narrow_histone", "E1b", PE),        # unseen (H2AFZ = H2A.Z), supported
    L("T05", "val", "one", "H3K9me3", "broad_histone", "E1c"),             # no_channel
    L("T06", "train", "five", "H3K27ac", "narrow_histone", "E5"),           # no_condition
    L("T07", "val", "big", "H2AFZ", "narrow_histone", "E3"),               # composite -> CL:3_CL:8, observed
    L("T08", "val", "two", "DNase-seq", "accessibility_dnase", "E2b"),     # unseen accessibility, exploratory
    L("T09", "val", "Caco-2", "H3K27ac", "narrow_histone", "E9b"),        # observed_by_name (EFO:5)
    L("C01", "val", "one", "chipseq-control", "control", "E9"),            # skipped
]
EXPERIMENTS = [{"experiment": e, "biosample_term_id": t} for e, t in
               [("E1", "CL:1"), ("E1b", "CL:1"), ("E1c", "CL:1"), ("E2", "CL:2"), ("E2b", "CL:2"),
                ("E3", "CL:3"), ("E4", "CL:4"), ("E5", "CL:6"), ("E9", "CL:1"),
                ("E9b", " cl_5 ")]]


def run_audit(ledger=LEDGER, observed=OBSERVED):
    return {r["eic_track"]: r for r in A.audit(ledger, EXPERIMENTS, A.Ag2Meta(observed, CONDITIONS, CHANNELS))}


def test_one_track_per_status():
    rows = run_audit()
    got = {t: r["status"] for t, r in rows.items()}
    assert got == {"T01": "observed_matched", "T02": "observed_other_read_type", "T03": "sibling_condition",
                   "T04": "unseen", "T05": "no_channel", "T06": "no_condition", "T07": "observed_matched",
                   "T08": "unseen", "T09": "observed_by_name"}
    assert {r["status"] for r in rows.values()} == set(A.STATUSES)
    assert [rows[t]["queryable"] for t in ("T01", "T02", "T03", "T04", "T05", "T06")] == [1, 1, 1, 1, 0, 0]
    assert rows["T01"]["api_label"] == "observed"
    assert rows["T04"]["api_label"] == "imputed_supported"
    assert rows["T08"]["api_label"] == "imputed_exploratory"
    assert rows["T05"]["api_label"] == rows["T06"]["api_label"] == ""
    assert (rows["T02"]["matched_channel"], rows["T02"]["otherend_channel"]) == ("K27ac_se", "K27ac_pe")
    assert rows["T04"]["matched_channel"] == "" and rows["T04"]["otherend_channel"] == "H2AZ_se"
    assert rows["T01"]["n_observed_channels_in_cell"] == 2 and rows["T06"]["n_observed_channels_in_cell"] == 0


def test_composite_rule_picks_most_encode_bulk_tracks():
    rows = run_audit()
    assert (rows["T07"]["condition"], rows["T07"]["condition_match"]) == ("CL:3_CL:8", "composite")
    assert (rows["T01"]["condition"], rows["T01"]["condition_match"]) == ("CL:1", "equal")
    # an equal condition wins over a composite with more bulk tracks
    assert (rows["T03"]["condition"], rows["T03"]["condition_match"]) == ("CL:4", "equal")
    assert rows["T06"]["condition_match"] == "none"


def test_other_read_type_is_not_unseen():
    rows = run_audit()
    assert rows["T02"]["status"] == "observed_other_read_type"
    assert rows["T02"]["status"] != "unseen"
    sets = {(s["split"], s["model"], s["track_set"]): s for s in A.fair_sets(list(rows.values()))}
    assert sets[("val", "final", "ag2_trained")]["n_tracks"] == 3          # T01, T02, T07
    assert sets[("val", "final", "unseen_by_both")]["n_tracks"] == 1       # T08
    assert sets[("blind", "final_tv", "unseen_by_both")]["per_assay"] == "H2AFZ:1"


def test_row_count_equals_non_control_rows(tmp_path):
    files = {}
    for name, rows in [("ledger", LEDGER), ("experiments", EXPERIMENTS), ("observed", OBSERVED),
                       ("conditions", CONDITIONS), ("channels", CHANNELS)]:
        files[name] = tmp_path / f"{name}.tsv"
        A.write_tsv(files[name], list(rows[0]), rows)
    out = tmp_path / "out"
    rc = A.main(["run", *[x for k, p in files.items() for x in (f"--{k}", str(p))], "--out", str(out)])
    assert rc == 0
    with open(out / "ag2_eic_audit.tsv") as fh:
        got = list(csv.DictReader(fh, delimiter="\t"))
    assert len(got) == sum(r["assay_class"] != "control" for r in LEDGER) == 9
    assert list(got[0]) == A.AUDIT_COLS
    with open(out / "fetch_conditions.tsv") as fh:
        fc = {r["condition"]: r for r in csv.DictReader(fh, delimiter="\t")}
    assert set(fc) == {"CL:1", "CL:2", "CL:3_CL:8", "CL:4", "EFO:5"}
    assert (out / "ag2_name_mismatches.tsv").exists()
    assert fc["CL:2"]["needs_accessibility"] == "1" and fc["CL:1"]["needs_accessibility"] == "0"
    assert (fc["CL:1"]["n_val_tracks"], fc["CL:1"]["n_blind_tracks"]) == ("1", "1")


def test_trained_on_accessibility_track_needs_accessibility():
    # CL:1 has an observed DNase-seq track: a validation DNase-seq track there is in the AG2-trained set,
    # and is still fetched (PI ruling 34). A sibling-condition accessibility track is not scored.
    extra = [L("T10", "val", "one", "DNase-seq", "accessibility_dnase", "E1d"),
             L("T11", "blind", "four", "DNase-seq", "accessibility_dnase", "E4b")]
    exps = EXPERIMENTS + [{"experiment": "E1d", "biosample_term_id": "CL:1"},
                          {"experiment": "E4b", "biosample_term_id": "CL:4"}]
    observed = OBSERVED + [obs("CL:4_CL:7", "DNASE", "", "", "dnase")]
    rows = A.audit(LEDGER + extra, exps, A.Ag2Meta(observed, CONDITIONS, CHANNELS))
    st = {r["eic_track"]: r["status"] for r in rows}
    assert st["T10"] == "observed_matched" and st["T11"] == "sibling_condition"
    fc = {r["condition"]: r for r in A.fetch_conditions(rows)}
    assert fc["CL:1"]["needs_accessibility"] == 1
    assert fc["CL:2"]["needs_accessibility"] == 1                # unseen DNase-seq (T08)
    assert fc["CL:4"]["needs_accessibility"] == 0                # sibling accessibility track is not scored


def test_unmapped_mark_is_an_error():
    bad = LEDGER + [L("T99", "val", "one", "H3K99xx", "narrow_histone", "E1")]
    with pytest.raises(A.AuditError, match="unmapped mark"):
        run_audit(ledger=bad)


def test_missing_term_is_an_error():
    bad = LEDGER + [L("T98", "val", "one", "H3K27ac", "narrow_histone", "E_missing")]
    with pytest.raises(A.AuditError, match="biosample_term_id"):
        run_audit(ledger=bad)


def _meta(observed=OBSERVED, channels=CHANNELS):
    return A.Ag2Meta(observed, CONDITIONS, channels)


def test_fix1_observed_key_without_channel_is_an_error():
    assert run_audit()["T01"]["status"] == "observed_matched"            # H3K27AC matched case-insensitively
    with pytest.raises(A.AuditError, match="no row in the channels table"):
        _meta(observed=OBSERVED + [obs("CL:2", "CHIP_HISTONE", "H3K4me1", "single", "K4me1_se")])
    with pytest.raises(A.AuditError, match="endedness"):
        _meta(observed=OBSERVED + [obs("CL:2", "CHIP_HISTONE", "H3K27ac", "both", "K27ac_se")])
    bad_ch = CHANNELS + [{"output_type": "ATAC", "channel_index": "0", "channel": "atac", "target": "",
                          "endedness": "SE"}]
    with pytest.raises(A.AuditError, match="endedness"):
        _meta(channels=bad_ch)


def test_fix2_condition_without_observed_track_is_no_condition():
    led = [L("T20", "val", "ten", "H3K27ac", "narrow_histone", "E20")]
    exps = [{"experiment": "E20", "biosample_term_id": "CL:10"}]
    (r,) = A.audit(led, exps, _meta())
    assert (r["condition"], r["status"], r["queryable"]) == ("CL:10", "no_condition", 0)
    assert A.fetch_conditions([r]) == []


def test_fix3_name_match_with_other_id_is_trained_on():
    rows = run_audit()
    r = rows["T09"]
    assert (r["status"], r["condition"], r["condition_match"], r["queryable"]) == \
        ("observed_by_name", "EFO:5", "name", 1)
    assert all(r not in v for v in A.set_members(list(rows.values())).values())
    led = LEDGER + [L("T21", "val", "eins", "H3K27ac", "narrow_histone", "E1")]
    mm = A.name_mismatches(A.audit(led, EXPERIMENTS, _meta()), _meta())
    kinds = {(m["cell"], m["kind"], m["condition"]) for m in mm}
    assert kinds == {("Caco-2", "name_match_id_differs", "EFO:5"), ("eins", "id_match_name_differs", "CL:1")}


def test_fix4_curies_are_normalised_and_malformed_ids_fail():
    assert A.norm_curie(" efo_0001 ") == "EFO:0001"
    assert run_audit()["T09"]["biosample_term_id"] == "CL:5"
    with pytest.raises(A.AuditError, match="malformed"):
        A.audit(LEDGER, EXPERIMENTS + [{"experiment": "E1", "biosample_term_id": "CL-1"}], _meta())


def test_train_and_final_tv_validation_tracks_stay_out_of_every_set():
    led = LEDGER + [L("T22", "train", "one", "H2AFZ", "narrow_histone", "E1b"),     # unseen, but trained on
                    L("T23", "train", "one", "H3K27ac", "narrow_histone", "E1")]    # ag2-trained, trained on
    rows = A.audit(led, EXPERIMENTS, _meta())
    assert {r["eic_track"]: r["status"] for r in rows}["T22"] == "unseen"
    for (split, model, _), members in A.set_members(rows).items():
        assert all(m["role"] == split != "train" for m in members)
        if model in ("final_tv", "published_eic"):
            assert all(m["role"] == "blind" for m in members)
    assert any(m["role"] == "val" for m in A.set_members(rows)[("val", "final", "unseen_by_both")])


def test_non_encode_observed_row_counts_as_trained():
    observed = OBSERVED + [obs("CL:2", "CHIP_HISTONE", "H2A.Z", "single", "H2AZ_se", source="roadmap")]
    led = [L("T24", "val", "two", "H2AFZ", "narrow_histone", "E2")]
    (r,) = A.audit(led, EXPERIMENTS, _meta(observed=observed))
    assert r["status"] == "observed_matched"
    assert r in A.set_members([r])[("val", "final", "ag2_trained")]
