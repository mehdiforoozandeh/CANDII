"""`python -m candi.store …` end to end, on a synthetic tree.

One pass through the CLI is what a SLURM array task actually runs, so the argument names are part
of the contract: renaming a flag here breaks `slurm/*.sh` with no test failure anywhere else.
"""
from __future__ import annotations

import json

import pytest

from candi.store import layout as L
from candi.store.cli import main

from tests.test_store_writer import CHROM_SIZES, default_rows, make_source_tree, write_csv


@pytest.fixture()
def tree(tmp_path):
    src = tmp_path / "src"
    make_source_tree(src, kinds=("counts", "peaks", "pval"))
    sizes = tmp_path / "chrom_sizes.json"
    sizes.write_text(json.dumps(CHROM_SIZES), encoding="utf-8")
    return {"src": src, "sizes": sizes, "corpus": tmp_path / "store" / "eic", "tmp": tmp_path}


def test_build_then_manifest_then_verify(tree, capsys):
    assert main([
        "build-biosample",
        "--source-root", str(tree["src"]),
        "--corpus-root", str(tree["corpus"]),
        "--chrom-sizes", str(tree["sizes"]),
        "--kinds", "counts,peaks,pval",
    ]) == 0
    assert L.kind_path(tree["corpus"], "T_SYNTH", "counts").is_file()

    csv_path = write_csv(tree["tmp"] / "eic_metadata.csv", default_rows())
    assert main([
        "build-manifest",
        "--corpus-root", str(tree["corpus"]),
        "--corpus", "eic",
        "--metadata-csv", str(csv_path),
        "--source-root", str(tree["src"]),
    ]) == 0
    manifest = json.loads(L.manifest_path(tree["corpus"]).read_text(encoding="utf-8"))
    assert manifest["corpus"] == "eic" and manifest["schema"] == 1

    assert main(["verify", "--corpus-root", str(tree["corpus"])]) == 0
    assert "OK" in capsys.readouterr().out


def test_verify_returns_nonzero_when_something_is_wrong(tree):
    main(["build-biosample", "--source-root", str(tree["src"]),
          "--corpus-root", str(tree["corpus"]), "--chrom-sizes", str(tree["sizes"]),
          "--kinds", "counts"])
    csv_path = write_csv(tree["tmp"] / "eic_metadata.csv", default_rows())
    main(["build-manifest", "--corpus-root", str(tree["corpus"]), "--corpus", "eic",
          "--metadata-csv", str(csv_path)])
    L.kind_path(tree["corpus"], "T_SYNTH", "counts").unlink()
    assert main(["verify", "--corpus-root", str(tree["corpus"])]) == 1


def test_several_metadata_csvs_are_accepted(tree):
    """This is how t5's `control_metadata.csv` joins the build — a repeated flag, nothing more."""
    main(["build-biosample", "--source-root", str(tree["src"]),
          "--corpus-root", str(tree["corpus"]), "--chrom-sizes", str(tree["sizes"]),
          "--kinds", "counts"])
    a = write_csv(tree["tmp"] / "eic_metadata.csv", default_rows())
    b = write_csv(tree["tmp"] / "control_metadata.csv",
                  ["T_SYNTH,chipseq-control,ENCBS001,ENCSR009,ENCFF-chipseq-control,GRCh38,"
                   "36,single-ended,Illumina HiSeq 2500,Synthetic Lab,9000000\n"])
    assert main(["build-manifest", "--corpus-root", str(tree["corpus"]), "--corpus", "eic",
                 "--metadata-csv", str(a), "--metadata-csv", str(b),
                 "--source-root", str(tree["src"])]) == 0
    manifest = json.loads(L.manifest_path(tree["corpus"]).read_text(encoding="utf-8"))
    ctrl = next(t for t in manifest["biosamples"]["T_SYNTH"]["tracks"]
                if t["assay"] == "chipseq-control")
    assert ctrl["depth"] == 9_000_000


def test_chrom_sizes_defaults_to_the_sibling_genome_dir(tree):
    gdir = L.corpus_genome_dir(tree["corpus"])
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "chrom_sizes.json").write_text(json.dumps(CHROM_SIZES), encoding="utf-8")
    assert main(["build-biosample", "--source-root", str(tree["src"]),
                 "--corpus-root", str(tree["corpus"]), "--kinds", "counts"]) == 0


def test_missing_chrom_sizes_is_a_clear_exit_not_a_traceback(tree):
    with pytest.raises(SystemExit, match="chrom-sizes"):
        main(["build-biosample", "--source-root", str(tree["src"]),
              "--corpus-root", str(tree["corpus"]), "--kinds", "counts"])


def test_build_genome_is_a_stub_that_points_at_t7(tree):
    """t7 owns `genome.py`. The stub must name the task, not fail with an AttributeError."""
    with pytest.raises(NotImplementedError, match="t7"):
        main(["build-genome"])
