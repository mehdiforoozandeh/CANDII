"""The two converters, checked against the three things that could silently corrupt a panel.

Run from the method directory (it is a standalone tool, not part of the `candi` package):

    PYTHONPATH=$PWD/../../src:$PWD python -m pytest competitors/chromimpute/tests -q

The three: the bin grid (an off-by-one shifts every score), the fairness filter (a `V_` cell in
`inputinfofile.txt` invalidates every number the method produces), and the directory name (a track
the bench cannot match to a declared pair is a track it drops).

A fourth: the training grid. `GenerateTrainData` samples anywhere in a chromosome it is given and
nowhere outside one, so the only thing that can be checked here is the grid — that every base the
jar is allowed to reach in training is a base Rule 2 allows.
"""
from __future__ import annotations

import gzip
import json
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
# the training grid — Rule 2's locus scope
# ---------------------------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[3]
PILOT = REPO / "configs" / "regime.eic_pilot.json"
NO_REGIONS = REPO / "configs" / "regime.eic_19.json"

#: The store's bin table for the chromosomes these tests reach, from BENCHMARK_DESIGN.md §3.
N_BINS = {"chr19": 2344704, "chr20": 2577766, "chr21": 1868399, "chr22": 2032738}


def _bed_intervals(path):
    """The BED read independently of `candi.store.regime`, so the check is not the code again."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        f = line.split("\t")
        out.append((f[0], int(f[1]), int(f[2]), f[3]))
    return out


def _bed(tmp_path, rows):
    p = tmp_path / "r.bed"
    p.write_text("".join(f"{c}\t{s}\t{e}\t{n}\n" for c, s, e, n in rows), encoding="utf-8")
    return p


def _regions_regime(tmp_path, rows, train_chroms):
    import hashlib
    bed = _bed(tmp_path, rows)
    regime = tmp_path / "regime.json"
    regime.write_text(json.dumps({
        "train_chroms": list(train_chroms),
        "regions": {"bed": bed.name,
                    "sha256": hashlib.sha256(bed.read_bytes()).hexdigest(),
                    "policy": "contain"},
    }), encoding="utf-8")
    return regime


def test_a_regime_without_a_regions_block_declares_no_region_scope():
    """`region_scope` is the BED reader and nothing else; absent a BED it has nothing to say."""
    assert prepare.region_scope(prepare.load_json(NO_REGIONS), NO_REGIONS) == []


def test_no_sampled_training_location_can_fall_on_an_evaluation_chromosome():
    """The requirement, in both live regimes. `GenerateTrainData` samples anywhere in a declared
    chromosome, so the guarantee has to be that the eval chromosomes are not declared to it —
    not that the sampler behaves. Rule 2 (§2) is about the loci the predictors are fit on."""
    for path in (NO_REGIONS, PILOT):
        regime = prepare.load_json(path)
        scope = prepare.training_scope(regime, path, N_BINS)
        assert scope, path.name
        assert not {c for _, c, _, _ in scope} & set(regime["eval_chroms"]), path.name


def test_the_training_grid_is_the_regimes_train_chroms_without_a_bed():
    """The PI ruling of 2026-09-01: the sampler goes where the regime says the transferable
    parameters may be fit, which without a BED is `train_chroms` whole."""
    regime = prepare.load_json(NO_REGIONS)
    assert prepare.training_scope(regime, NO_REGIONS, N_BINS) == [("chr19", "chr19", 0, 2344704)]
    assert regime["train_chroms"] == ["chr19"]


def test_a_bed_narrows_the_training_grid_further_but_never_widens_it():
    """A `regions` regime restricts the same scope; it never adds a locus `train_chroms` excludes."""
    regime = prepare.load_json(PILOT)
    scope = prepare.training_scope(regime, PILOT, N_BINS)
    assert scope == prepare.region_scope(regime, PILOT)
    assert {c for _, c, _, _ in scope} <= set(regime["train_chroms"])


def test_widening_the_training_grid_does_not_widen_the_compendium():
    """Rule 1 does not move when Rule 2's scope does. The grid decides which BASES are written;
    `training_tracks` alone decides which TRACKS are, and it is the only gate on the scored ones."""
    tracks = prepare.training_tracks(_manifest(), _regime())
    for scope in ([("chr19", "chr19", 0, 10)], [("R1", "chr1", 0, 10), ("R2", "chr2", 0, 10)]):
        names = {prepare.signal_filename(s, m) for s, m in tracks
                 for _ in prepare.signal_slices(["chr20"], {"chr20": 10}, scope)}
        assert not {n for n in names if n.startswith(("V_", "B_"))}
        assert {n.split(".")[0] for n in names} == {"T_K562", "T_H9"}


def test_a_regime_that_names_no_training_loci_is_refused(tmp_path):
    """Silence is the failure mode this replaces: a regime with no scope used to fall through to
    the eval chromosomes, which is exactly what Rule 2 forbids."""
    regime = tmp_path / "regime.json"
    regime.write_text(json.dumps({"train_chroms": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no locus scope"):
        prepare.training_scope(prepare.load_json(regime), regime, N_BINS)


def test_a_training_chromosome_the_store_does_not_carry_is_named(tmp_path):
    regime = tmp_path / "regime.json"
    regime.write_text(json.dumps({"train_chroms": ["chrZZ"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="chrZZ"):
        prepare.training_scope(prepare.load_json(regime), regime, N_BINS)


def test_every_training_location_the_jar_can_reach_lies_inside_the_bed():
    """The requirement. The jar samples anywhere in a declared chromosome, so the guarantee has to
    be that no declared training base is outside a region — not that the sampler behaves."""
    regime = prepare.load_json(PILOT)
    scope = prepare.region_scope(regime, PILOT)
    bed = _bed_intervals(REPO / "configs" / regime["regions"]["bed"])
    for name, chrom, first, stop in scope:
        inside = [(s, e) for c, s, e, n in bed if c == chrom and n == name]
        assert len(inside) == 1, name
        s, e = inside[0]
        assert s <= first * prepare.RESOLUTION
        assert stop * prepare.RESOLUTION <= e


def test_the_bed_keeps_its_regions_on_the_evaluation_chromosomes_and_the_scope_drops_them():
    """Rule 2's cut is the regime's chromosome list, not the BED — four Pilot Regions sit on
    chr20/21/22 and the BED keeps all 44 so it stays correct for any other split (§3.1)."""
    regime = prepare.load_json(PILOT)
    scope = prepare.region_scope(regime, PILOT)
    assert not {c for _, c, _, _ in scope} & set(regime["eval_chroms"])
    assert len(_bed_intervals(REPO / "configs" / regime["regions"]["bed"])) == 44


def test_the_pilot_training_scope_is_the_one_section_3_1_pins():
    scope = prepare.region_scope(prepare.load_json(PILOT), PILOT)
    assert len(scope) == 40
    assert sum(stop - first for _, _, first, stop in scope) == 1_023_489


def test_containment_is_counted_the_same_way_the_window_sampler_counts_it():
    """CANDI's loader and this share one rule and must not drift; `bin_spans` is the authority."""
    from candi.store.regime import RegionSet
    regime = prepare.load_json(PILOT)
    rs = RegionSet.from_obj(regime["regions"], base=REPO / "configs")
    scope = prepare.region_scope(regime, PILOT)
    for chrom in {c for _, c, _, _ in scope}:
        mine = sorted((f, s) for _, c, f, s in scope if c == chrom)
        assert mine == sorted(rs.bin_spans(chrom, prepare.RESOLUTION))


def test_a_region_edge_off_the_bin_grid_loses_its_partial_bin(tmp_path):
    """25,588,197 bp is not divisible by 25, so the scope is a containment count and a bin only
    half-covered by a region is not a training location."""
    regime = _regions_regime(tmp_path, [("chr1", 1010, 2490, "R1")], ["chr1"])
    assert prepare.region_scope(prepare.load_json(regime), regime) == [("R1", "chr1", 41, 99)]


def test_the_training_grid_is_anchored_at_chromosome_bin_zero(tmp_path):
    """§3.1 rejected re-anchoring the tiling per region, so a region's bins keep the chromosome's
    own indices and two regions on one chromosome cannot claim the same bin number."""
    regime = _regions_regime(tmp_path, [("chr1", 1000, 2000, "R1"), ("chr1", 5000, 6000, "R2")],
                             ["chr1"])
    assert prepare.region_scope(prepare.load_json(regime), regime) == [
        ("R1", "chr1", 40, 80), ("R2", "chr1", 200, 240)]


def test_a_region_scope_with_nothing_on_the_training_chromosomes_is_refused(tmp_path):
    regime = _regions_regime(tmp_path, [("chr21", 0, 1000, "R1")], ["chr1"])
    with pytest.raises(ValueError, match="no region on train_chroms"):
        prepare.region_scope(prepare.load_json(regime), regime)


def test_a_region_named_after_a_chromosome_is_refused(tmp_path):
    """Each region becomes a declared chromosome, so a name that is already one would make the
    training grid and the apply grid write to the same converted file."""
    regime = _regions_regime(tmp_path, [("chr1", 0, 1000, "chr1")], ["chr1"])
    with pytest.raises(ValueError, match="collisions"):
        prepare.region_scope(prepare.load_json(regime), regime)


def test_the_declared_region_length_makes_convert_emit_exactly_the_contained_bins(tmp_path):
    """The same `n_bins*25` trap as a real chromosome — one extra bin here is a training location
    outside the BED."""
    scope = prepare.region_scope(prepare.load_json(PILOT), PILOT)
    bins = {name: stop - first for name, _, first, stop in scope}
    path = prepare.write_chrominfo(tmp_path / "chrominfo.train.txt", bins, sorted(bins))
    for line in path.read_text().splitlines():
        name, length = line.split("\t")
        assert (int(length) - 1) // prepare.RESOLUTION + 1 == bins[name]


def test_the_signal_writer_covers_the_apply_grid_and_the_training_grid_and_nothing_else():
    """One list drives the writer, so a locus declared in `chrominfo.train.txt` and never written
    as a bedgraph — the failure `Convert` reports as a warning and skips — cannot happen."""
    scope = prepare.region_scope(prepare.load_json(PILOT), PILOT)
    slices = prepare.signal_slices(["chr20", "chr21"], {"chr20": 100, "chr21": 200}, scope)
    assert slices[:2] == [("chr20", "chr20", 0, 100), ("chr21", "chr21", 0, 200)]
    assert slices[2:] == scope
    assert len({name for name, _, _, _ in slices}) == len(slices)


def test_the_signal_writer_covers_a_training_chromosome_the_apply_grid_does_not_name():
    """The training chromosome needs its own bedgraphs — `Convert` skips a missing input with a
    warning, so an unwritten chr19 would leave the predictors fitted on nothing at all."""
    scope = prepare.training_scope(prepare.load_json(NO_REGIONS), NO_REGIONS, N_BINS)
    slices = prepare.signal_slices(["chr20", "chr21", "chr22"], N_BINS, scope)
    assert ("chr19", "chr19", 0, N_BINS["chr19"]) in slices
    assert [name for name, _, _, _ in slices] == ["chr20", "chr21", "chr22", "chr19"]


def test_a_chromosome_on_both_grids_is_written_once():
    """The two grids may name the same chromosome; one bedgraph serves both, and writing it twice
    would be the same bytes at twice the cost."""
    slices = prepare.signal_slices(["chr19", "chr20"], N_BINS,
                                   [("chr19", "chr19", 0, N_BINS["chr19"])])
    assert [name for name, _, _, _ in slices] == ["chr19", "chr20"]


# ---------------------------------------------------------------------------------------------
# which chrominfo each jar command is handed — `stage.sh`, driven against a recording `java`
# ---------------------------------------------------------------------------------------------

STAGE_SH = REPO / "competitors" / "chromimpute" / "slurm" / "stage.sh"
SCORE_SH = REPO / "competitors" / "chromimpute" / "slurm" / "score.sh"


def _shim(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    p.chmod(0o755)


def _run_stage(tmp_path, stage, item, *, write_train_grid=True):
    """Run one `stage.sh` stage with `java` replaced by a recorder. Returns (proc, arg lines)."""
    import subprocess
    run, bin_dir = tmp_path / "run", tmp_path / "bin"
    (run / "input").mkdir(parents=True)
    (run / "lists").mkdir()
    bin_dir.mkdir()
    log = tmp_path / "java.log"
    _shim(bin_dir, "java", f'printf "%s\\n" "$*" >> {log}\n')
    _shim(bin_dir, "module", "true\n")
    (run / "input" / "chrominfo.txt").write_text(
        "chr20\t100\nchr21\t100\nchr22\t100\n", encoding="utf-8")
    if write_train_grid:
        (run / "input" / "chrominfo.train.txt").write_text("chr19\t100\n", encoding="utf-8")
    (run / "input" / "inputinfofile.txt").write_text(
        "T_K562\tH3K4me3\tT_K562.H3K4me3.bedgraph.gz\n", encoding="utf-8")
    (run / "lists" / f"{stage}.txt").write_text(item + "\n", encoding="utf-8")
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path),
        "CI_STAGE": stage, "CI_RUN": str(run), "CI_REGIME": str(NO_REGIONS),
        "CI_CHROMS": "chr20,chr21,chr22", "CI_PY": sys.executable,
        "CI_JAR": str(tmp_path / "ChromImpute.jar"), "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_ID": "1", "CI_KEEP_BEDGRAPH": "1",
    }
    proc = subprocess.run(["bash", str(STAGE_SH)], env=env, capture_output=True, text=True)
    lines = log.read_text().splitlines() if log.exists() else []
    return proc, lines


def test_generate_train_data_is_handed_the_training_grid_and_never_the_apply_grid():
    """The requirement, at the only place it can be enforced: which file reaches the sampler."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        proc, lines = _run_stage(Path(d), "gtd", "H3K4me3")
    assert proc.returncode == 0, proc.stderr
    gtd = [ln for ln in lines if "GenerateTrainData" in ln]
    assert len(gtd) == 1, lines
    assert "chrominfo.train.txt" in gtd[0]
    assert "input/chrominfo.txt" not in gtd[0]


def test_the_correlation_table_is_computed_on_the_training_grid_too():
    """`ComputeGlobalDist` writes one sample ranking reused at every predicted position, so it is
    a transferable parameter and belongs on the training loci with the predictors."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        proc, lines = _run_stage(Path(d), "dist", "H3K4me3")
    assert proc.returncode == 0, proc.stderr
    dist = [ln for ln in lines if "ComputeGlobalDist" in ln]
    assert len(dist) == 1 and "chrominfo.train.txt" in dist[0]


def test_apply_still_runs_on_every_evaluation_chromosome():
    """§2 Rule 2 names per-position adaptation on the scored chromosomes as inference and open to
    every method, and lists ChromImpute's neighbour features. Moving the sampler must not move it."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        proc, lines = _run_stage(Path(d), "apply", "T_K562\tH3K4me3")
    assert proc.returncode == 0, proc.stderr
    applied = [ln for ln in lines if " Apply " in f" {ln} "]
    assert sorted(ln.split("-c ")[1].split()[0] for ln in applied) == ["chr20", "chr21", "chr22"]
    assert all("input/chrominfo.txt" in ln for ln in applied)
    assert not any("chrominfo.train.txt" in ln for ln in applied)


def test_convert_prepares_the_training_chromosome_as_well_as_the_evaluation_ones():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        proc, lines = _run_stage(Path(d), "convert", "H3K4me3")
    conv = [ln for ln in lines if "Convert" in ln]
    assert sorted(ln.split("-c ")[1].split()[0] for ln in conv) == \
        ["chr19", "chr20", "chr21", "chr22"]


def test_a_missing_training_grid_refuses_instead_of_falling_back_to_the_apply_grid():
    """The fallback was the bug: it silently fitted the predictors on the scored chromosomes."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        proc, lines = _run_stage(Path(d), "gtd", "H3K4me3", write_train_grid=False)
    assert proc.returncode == 2
    assert "REFUSING" in proc.stderr
    assert not lines


def test_a_void_sigma_table_is_refused_rather_than_scored():
    """§12.2 voids every sigma this tree can produce — `fit_sigma.fit` pools residuals against V_
    truth. Avocado, eDICE and Lavawizard refuse to fit one; this script refuses to use one."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "sigma.json").write_text("{}", encoding="utf-8")
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp), "CI_RUN": str(tmp),
               "CI_SIGMA": str(tmp / "sigma.json")}
        proc = subprocess.run(["bash", str(SCORE_SH)], env=env, capture_output=True, text=True)
    assert proc.returncode == 3
    assert "REFUSING" in proc.stderr


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


# ---------------------------------------------------------------------------------------------
# the §6.1 sigma-table
# ---------------------------------------------------------------------------------------------


def _scores(rows):
    """rows: (assay, mse, n_points) -> a minimal scores json shaped like bench.external's."""
    return {"provenance": {"method": "ChromImpute"},
            "per_track": {f"T_x|V_x|{a}#{i}": {"pval": {
                "assay": a, "mse": m, "n_points": n, "signal_target_transform": "none"}}
                for i, (a, m, n) in enumerate(rows)}}


def test_sigma_is_the_root_of_the_pooled_mean_squared_residual():
    import fit_sigma
    out = fit_sigma.fit(_scores([("H3K4me3", 4.0, 100), ("H3K4me3", 16.0, 100)]))
    assert out["H3K4me3"]["mse_pooled"] == pytest.approx(10.0)
    assert out["H3K4me3"]["sigma"] == pytest.approx(10.0 ** 0.5)
    assert out["H3K4me3"]["n_tracks"] == 2


def test_sigma_pooling_is_weighted_by_bins_not_by_tracks():
    import fit_sigma
    out = fit_sigma.fit(_scores([("H3K9me3", 1.0, 900), ("H3K9me3", 11.0, 100)]))
    assert out["H3K9me3"]["mse_pooled"] == pytest.approx(2.0)


def test_sigma_refuses_a_track_scored_through_a_transform():
    import fit_sigma
    s = _scores([("H3K4me3", 4.0, 100)])
    next(iter(s["per_track"].values()))["pval"]["signal_target_transform"] = "arcsinh"
    with pytest.raises(ValueError, match="raw -log10 p"):
        fit_sigma.fit(s)


def test_sigma_table_has_the_three_fields_bench_external_reads(tmp_path):
    import json
    import fit_sigma
    src = tmp_path / "s.json"
    src.write_text(json.dumps(_scores([("H3K4me3", 4.0, 100)])))
    out = tmp_path / "sigma.json"
    fit_sigma.main([str(src), "--out", str(out)])
    obj = json.loads(out.read_text())
    assert obj["method"] == "ChromImpute"
    assert obj["fitted_on"] == "regime.eic_val eval_pairs"
    assert obj["sigma"]["H3K4me3"] == pytest.approx(2.0)


def test_sigma_table_is_accepted_by_the_bench_reader(tmp_path):
    """The two sides of the boundary must agree on the §4.2 shape, not just on our reading of it."""
    import json
    import fit_sigma
    from candi.bench.external import read_sigma_table
    src = tmp_path / "s.json"
    src.write_text(json.dumps(_scores([("H3K4me3", 4.0, 100), ("H3K9me3", 9.0, 100)])))
    out = tmp_path / "sigma.json"
    fit_sigma.main([str(src), "--out", str(out)])
    table = read_sigma_table(out)
    assert table["sigma"]["H3K9me3"] == pytest.approx(3.0)
