"""The two converters, checked against the three things that could silently corrupt a panel.

Run from the method directory (it is a standalone tool, not part of the `candi` package):

    PYTHONPATH=$PWD/../../src:$PWD python -m pytest competitors/chromimpute/tests -q

The three: the bin grid (an off-by-one shifts every score), the fairness filter (a `V_` cell in
`inputinfofile.txt` invalidates every number the method produces), and the directory name (a track
the bench cannot match to a declared pair is a track it drops).
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collect  # noqa: E402
import prepare  # noqa: E402


# ---------------------------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------------------------


def test_chrominfo_length_makes_convert_emit_exactly_n_bins(tmp_path):
    """`Convert` writes `(len-1)//25+1` bins; at `len = n_bins*25` that is `n_bins`."""
    n_bins = {"chr21": 1868399, "chr19": 2344704}
    path = prepare.write_chrominfo(tmp_path / "chrominfo.txt", n_bins, ["chr21", "chr19"])
    for line in path.read_text().splitlines():
        chrom, length = line.split("\t")
        assert (int(length) - 1) // prepare.RESOLUTION + 1 == n_bins[chrom]


def test_chrominfo_names_the_chromosome_it_cannot_find(tmp_path):
    with pytest.raises(ValueError, match="chrZZ"):
        prepare.write_chrominfo(tmp_path / "c.txt", {"chr21": 10}, ["chrZZ"])


def test_bedgraph_tiles_the_grid_without_gap_or_overlap():
    v = np.array([0.0, 0.0, 1.5, 1.5, 1.5, 0.25, 0.0], dtype=np.float32)
    lines = list(prepare.bedgraph_lines(v, "chr21"))
    spans = [(int(a), int(b)) for _, a, b, _ in (ln.split("\t") for ln in lines)]
    assert spans[0][0] == 0
    assert spans[-1][1] == v.size * prepare.RESOLUTION
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end == start


def test_bedgraph_round_trips_the_signal_at_the_declared_precision():
    rng = np.random.default_rng(0)
    v = np.where(rng.random(4000) < 0.6, 0.0, rng.gamma(2.0, 3.0, 4000)).astype(np.float32)
    lines = list(prepare.bedgraph_lines(v, "chr21", decimals=4))
    back = np.zeros(v.size)
    for ln in lines:
        _, a, b, x = ln.split("\t")
        back[int(a) // prepare.RESOLUTION:int(b) // prepare.RESOLUTION] = float(x)
    assert np.allclose(back, np.round(v.astype(np.float64), 4), atol=0)


def test_bedgraph_collapses_equal_neighbours():
    v = np.zeros(1000, dtype=np.float32)
    v[500] = 7.0
    assert len(list(prepare.bedgraph_lines(v, "chr21"))) == 3


def test_bedgraph_refuses_non_finite():
    v = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        list(prepare.bedgraph_lines(v, "chr21"))


def test_write_bedgraph_is_gzip_and_leaves_no_part_file(tmp_path):
    p = tmp_path / "chr21_T_K562.H3K4me3.bedgraph.gz"
    n = prepare.write_bedgraph(p, np.array([1.0, 1.0, 2.0], dtype=np.float32), "chr21")
    assert n == 2
    assert gzip.open(p, "rt").read().splitlines()[0].split("\t")[0] == "chr21"
    assert not list(tmp_path.glob("*.part"))


# ---------------------------------------------------------------------------------------------
# fairness (§6.2) and the target set
# ---------------------------------------------------------------------------------------------


def _manifest():
    def rec(assays):
        return {"kinds": ["pval"], "tracks": [{"assay": a} for a in assays]}
    return {"biosamples": {
        "T_K562": rec(["H3K4me3", "H3K27ac", "chipseq-control"]),
        "T_H9": rec(["H3K4me3"]),
        "V_K562": rec(["H3K4me3", "H3K9me3", "chipseq-control"]),
        "B_K562": rec(["H3K4me1"]),
    }}


def _regime():
    return {
        "assays": ["H3K4me3", "H3K27ac", "H3K9me3", "H3K4me1"],
        "biosamples": {"train": ["T_K562", "T_H9"], "eval": ["V_K562"]},
        "eval_pairs": [["T_K562", "V_K562"]],
    }


def test_training_tracks_admit_training_cells_only():
    tracks = prepare.training_tracks(_manifest(), _regime())
    assert {s for s, _ in tracks} == {"T_K562", "T_H9"}


def test_training_tracks_drop_the_control_column():
    assert "chipseq-control" not in {m for _, m in prepare.training_tracks(_manifest(), _regime())}


def test_training_tracks_refuse_a_non_training_prefix():
    regime = _regime()
    regime["biosamples"]["train"] = ["T_K562", "V_K562"]
    with pytest.raises(ValueError, match="V_K562"):
        prepare.training_tracks(_manifest(), regime)


def test_targets_are_the_assays_the_pair_holds_out():
    assert prepare.impute_targets(_manifest(), _regime()) == [("T_K562", "V_K562", "H3K9me3")]


def test_targets_json_round_trip(tmp_path):
    targets = [("T_K562", "V_K562", "H3K9me3"), ("T_H9", "V_H9", "H3K4me1")]
    p = prepare.write_targets(tmp_path / "targets.tsv", targets)
    assert prepare.read_targets(p) == targets


def test_pilot_subset_is_one_target_per_mark_and_deterministic():
    targets = [("T_a", "V_a", "H3K4me3"), ("T_b", "V_b", "H3K4me3"),
               ("T_c", "V_c", "H3K9me3"), ("T_d", "V_d", "H3K27ac")]
    sub = prepare.pilot_subset(targets, 2)
    assert len({a for _, _, a in sub}) == 2
    assert sub == prepare.pilot_subset(list(reversed(targets)), 2)
    # widest mark first: H3K4me3 carries two targets, the others one each.
    assert "H3K4me3" in {a for _, _, a in sub}


def test_pilot_subset_caps_at_the_number_of_marks():
    targets = [("T_a", "V_a", "H3K4me3"), ("T_b", "V_b", "H3K9me3")]
    assert len(prepare.pilot_subset(targets, 20)) == 2


def test_inputinfo_is_three_tab_columns(tmp_path):
    p = prepare.write_inputinfo(tmp_path / "i.txt", [("T_K562", "H3K4me3")])
    line = p.read_text().splitlines()[0]
    assert line.split("\t") == ["T_K562", "H3K4me3", "T_K562.H3K4me3.bedgraph.gz"]


# ---------------------------------------------------------------------------------------------
# the return leg
# ---------------------------------------------------------------------------------------------


def _wig(path: Path, values, *, browser_header=True):
    with gzip.open(path, "wt") as fh:
        if browser_header:
            fh.write("track type=wiggle_0 name=T_K562_H3K4me3_imputed\n")
        fh.write("fixedStep  chrom=chr21 start=1 step=25 span=25\n")
        fh.writelines(f"{v}\n" for v in values)


@pytest.mark.parametrize("browser_header", [True, False])
def test_read_wig_skips_however_many_headers_apply_wrote(tmp_path, browser_header):
    p = tmp_path / "chr21_impute.T_K562.H3K4me3.wig.gz"
    _wig(p, [0.0, 1.25, 3.5], browser_header=browser_header)
    assert collect.read_wig(p, 3).tolist() == [0.0, 1.25, 3.5]


def test_read_wig_refuses_the_wrong_bin_count(tmp_path):
    p = tmp_path / "w.gz"
    _wig(p, [0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="n_bins\\*25"):
        collect.read_wig(p, 3)


def test_read_wig_refuses_variable_step(tmp_path):
    p = tmp_path / "w.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("variableStep chrom=chr21 span=25\n1\t3.0\n")
    with pytest.raises(ValueError, match="variableStep"):
        collect.read_wig(p, 1)


def test_read_wig_returns_float32(tmp_path):
    p = tmp_path / "w.gz"
    _wig(p, [1.0])
    assert collect.read_wig(p, 1).dtype == np.float32


def test_track_dirname_matches_the_bench_contract():
    """§4.1 is one string; the two sides of the boundary must agree on it letter for letter."""
    from candi.bench.external import track_dirname as bench_dirname
    from candi.bench.harness import Pair
    pair = Pair(input_biosample="T_K562", target_biosample="V_K562")
    assert collect.track_dirname("T_K562", "V_K562", "H3K4me3") == bench_dirname(pair, "H3K4me3")


def test_write_track_writes_signal_mu_only(tmp_path):
    d = collect.write_track(tmp_path, "T_K562", "V_K562", "H3K4me3",
                            {"chr21": np.arange(4, dtype=np.float32)})
    with np.load(d / "chr21.npz") as z:
        assert list(z.keys()) == ["signal_mu"]
        assert z["signal_mu"].dtype == np.float32


def test_manifest_carries_the_method_the_bench_requires(tmp_path):
    import json
    collect.write_manifest(tmp_path, version="1.0.5", notes="pilot")
    obj = json.loads((tmp_path / "manifest.json").read_text())
    assert obj["method"] == "ChromImpute"
    assert obj["arms"] == ["pval"]


def test_apply_output_name_is_splittable_on_our_separator():
    name = collect.apply_output_name("T_upper_lobe_of_left_lung", "DNase-seq")
    assert name.split(".")[1] == "T_upper_lobe_of_left_lung"
    assert name.split(".")[2] == "DNase-seq"
