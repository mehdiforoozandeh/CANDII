"""XRd (Amendment 3): RNA-seq input tool on a mocked ENCODE API reply and synthetic bigWig arrays."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

for _p in (Path(__file__).resolve().parents[1] / "tools",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import rna_inputs as R  # noqa: E402

h5py = pytest.importorskip("h5py")


def _f(acc, ot, ann="V29", reps=(1,), fmt="bigWig", pref=None, size=100, md5=None):
    return {"accession": acc, "output_type": ot, "genome_annotation": ann, "assembly": "GRCh38",
            "biological_replicates": list(reps), "file_format": fmt, "preferred_default": pref,
            "file_size": size, "md5sum": md5 or acc.lower().ljust(32, "0")[:32],
            "status": "released", "cloud_metadata": {"url": f"https://s3/{acc}.{fmt}"}}


#: E1: stranded, V24 and V29; E2: unstranded only, V24 only; E3: perturbed; E4: no gene quant
FILES = {
    "ENCSR000EA1": [_f("ENCFF1P29", R.OT_PLUS), _f("ENCFF1M29", R.OT_MINUS),
                    _f("ENCFF1P24", R.OT_PLUS, "V24"), _f("ENCFF1M24", R.OT_MINUS, "V24"),
                    _f("ENCFF1G24", R.OT_GENE_QUANT, "V24", fmt="tsv", pref=True),
                    _f("ENCFF1G29", R.OT_GENE_QUANT, "V29", fmt="tsv"),
                    _f("ENCFF1G29B", R.OT_GENE_QUANT, "V29", reps=(2,), fmt="tsv", pref=True),
                    _f("ENCFF1ALL", "plus strand signal of all reads")],
    "ENCSR000EA2": [_f("ENCFF2U24", R.OT_UNSTRANDED, "V24", size=300),
                    _f("ENCFF2G24", R.OT_GENE_QUANT, "V24", fmt="tsv")],
    "ENCSR000EA4": [_f("ENCFF4P29", R.OT_PLUS), _f("ENCFF4M29", R.OT_MINUS)],
    "ENCSR000EA5": [_f("ENCFF5P29", R.OT_PLUS), _f("ENCFF5M29", R.OT_MINUS),
                    _f("ENCFF5G29", R.OT_GENE_QUANT, "V29", fmt="tsv")],
}
EXPS = {"ENCSR000EA1": {}, "ENCSR000EA2": {}, "ENCSR000EA3": {"perturbed": True},
        "ENCSR000EA4": {}, "ENCSR000EA5": {}}

CENSUS = [
    {"cell": "C01", "eic_name": "cell_a", "best_exp": "ENCSR000EA1",
     "same_donor_or_biosample": "yes (same biosample)"},
    {"cell": "C02", "eic_name": "cell_b", "best_exp": "ENCSR000EA2",
     "same_donor_or_biosample": "no (term only, other donor/biosample)"},
    {"cell": "C03", "eic_name": "cell_c", "best_exp": "ENCSR000EA3",
     "same_donor_or_biosample": "yes (same donor)"},
    {"cell": "C04", "eic_name": "cell_d", "best_exp": "ENCSR000EA4",
     "same_donor_or_biosample": "yes (same donor)"},
    {"cell": "C05", "eic_name": "cell_e", "best_exp": "ENCSR000EA5",
     "same_donor_or_biosample": "yes (same donor)"},
    {"cell": "C06", "eic_name": "cell_f", "best_exp": "", "same_donor_or_biosample": "n/a"},
]


def fake_get_json(url):
    if "/search/" in url:
        exp = url.split("/experiments/")[1].split("/")[0]
        return {"@graph": FILES.get(exp, [])}
    exp = url.split("/experiments/")[1].split("/")[0]
    e = {"status": "released", "perturbed": False, "assay_title": "polyA plus RNA-seq"}
    e.update(EXPS[exp])
    return e


@pytest.fixture
def listed(monkeypatch):
    monkeypatch.setattr(R, "get_json", fake_get_json)
    return R.list_cells(CENSUS)


def test_list_rules(listed):
    rows, missing = listed
    by = {r["cell"]: r for r in rows}
    assert sorted(by) == ["cell_a", "cell_b", "cell_e"]
    assert dict(missing) == {"cell_c": "ENCSR000EA3: perturbed",
                             "cell_d": "ENCSR000EA4: no GRCh38 gene quantifications",
                             "cell_f": "no RNA-seq experiment in the census"}
    a = by["cell_a"]
    # stranded, highest annotation version of the signal
    assert (a["stranded"], a["plus_bigwig"], a["minus_bigwig"]) == (1, "ENCFF1P29", "ENCFF1M29")
    # V29 is held by 2 cells (a, e), V24 by 2 (a, b): tie -> higher version, V29.
    # Within V29 the signal's replicate set (rep 1) beats preferred_default (rep 2).
    assert (a["gencode"], a["gene_quant"]) == ("V29", "ENCFF1G29")
    assert a["rna_match"] == "same" and a["cell_id"] == "C01"
    b = by["cell_b"]
    assert (b["stranded"], b["plus_bigwig"], b["minus_bigwig"]) == (0, "ENCFF2U24", "ENCFF2U24")
    assert (b["gencode"], b["rna_match"]) == ("V24", "other_donor")
    assert b["bytes"] == 300 + 100            # the unstranded file counted once
    assert a["bytes"] == 300


def test_majority_gencode_prefers_count_then_version():
    assert R.majority_gencode([{"V24"}, {"V24", "V29"}, {"V24"}]) == "V24"
    assert R.majority_gencode([{"V24"}, {"V29"}]) == "V29"


def test_fetch_refuses_above_cap_and_dedupes(listed, tmp_path):
    rows, _ = listed
    got = {}
    R.fetch(rows, tmp_path, max_gb=1.0, fetcher=lambda items: got.setdefault("items", items))
    accs = [it["accession"] for it in got["items"]]
    assert len(accs) == len(set(accs)) == 3 + 2 + 3
    assert all(it["md5"] and it["dest"].startswith(str(tmp_path)) for it in got["items"])
    with pytest.raises(SystemExit, match="refused"):
        R.fetch(rows, tmp_path, max_gb=1e-9, fetcher=lambda items: None)


SIZES = {"chr1": 100, "chr2": 60}


def fake_bigwig(values):
    """A bigWig as a per-base array, binned to the 25 bp exact mean like `bin_bedgraph`."""
    def reader(path, sizes, tmpdir):
        v = values[Path(path).stem]
        return {c: v[c][: (sizes[c] // 25) * 25].reshape(-1, 25).mean(axis=1).astype(np.float32)
                for c in sizes}
    return reader


def test_tracks_layout_and_unstranded_half(listed, tmp_path):
    rows, _ = listed
    rng = np.random.default_rng(0)
    base = {acc: {c: rng.gamma(1.0, 2.0, n) for c, n in SIZES.items()}
            for acc in ("ENCFF1P29", "ENCFF1M29", "ENCFF2U24", "ENCFF5P29", "ENCFF5M29")}
    base["ENCFF1M29"] = {c: -v for c, v in base["ENCFF1M29"].items()}   # negative minus strand
    out = tmp_path / "tracks"
    n = R.build_tracks(rows, SIZES, out, tmp_path / "files", tmp_path / "tmp",
                       reader=fake_bigwig(base))
    assert n == 3
    with h5py.File(out / "cell_a.h5", "r") as h:
        assert set(h) == {"plus", "minus"} and set(h["plus"]) == set(SIZES)
        assert h["plus/chr1"].dtype == np.float16 and h["plus/chr1"].shape == (4,)
        assert h["minus/chr2"].shape == (2,)
        want = np.abs(base["ENCFF1M29"]["chr1"].reshape(4, 25).mean(1))
        np.testing.assert_allclose(h["minus/chr1"][:], want, rtol=1e-3)
        assert h.attrs["minus_negative_sum"] < 0 and h.attrs["stranded"] == 1
    with h5py.File(out / "cell_b.h5", "r") as h:
        half = base["ENCFF2U24"]["chr1"].reshape(4, 25).mean(1) / 2
        np.testing.assert_allclose(h["plus/chr1"][:], half, rtol=1e-3)
        np.testing.assert_array_equal(h["plus/chr1"][:], h["minus/chr1"][:])
        assert h.attrs["stranded"] == 0


def _quant(path, tpm: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["gene_id\ttranscript_id(s)\tlength\teffective_length\texpected_count\tTPM\tFPKM"]
    lines += [f"{g}\tT\t1\t1\t1\t{v}\t0" for g, v in tpm.items()]
    path.write_text("\n".join(lines) + "\n")


def test_vector_gene_rule(listed, tmp_path):
    rows, _ = listed
    files = tmp_path / "files"
    # G1..G5 protein coding; G6 lncRNA; G7 absent from cell_b; spike-in and PAR_Y dropped
    q = {
        "cell_a": {"ENSG01.3": 0, "ENSG02.1": 10, "ENSG03.1": 5, "ENSG04.1": 1, "ENSG05.1": 3,
                   "ENSG06.1": 1000, "ENSG07.1": 1000, "ERCC-1": 9, "ENSG04.1_PAR_Y": 50},
        "cell_b": {"ENSG01.5": 100, "ENSG02.2": 10, "ENSG03.2": 0, "ENSG04.2": 1, "ENSG05.2": 3,
                   "ENSG06.2": 0},
        "cell_e": {"ENSG01.3": 50, "ENSG02.1": 10, "ENSG03.1": 5, "ENSG04.1": 1, "ENSG05.1": 3,
                   "ENSG06.1": 0, "ENSG07.1": 0},
    }
    for r in rows:
        _quant(R.local_path(r, "gene_quant", files), q[r["cell"]])
    gt = tmp_path / "types.tsv"
    gt.write_text("gene_id\tgene_type\n" + "".join(
        f"ENSG0{i}.9\tprotein_coding\n" for i in (1, 2, 3, 4, 5, 7)) + "ENSG06.9\tlncRNA\n")
    rec = R.build_vector(rows, gt, tmp_path / "root", files, n=2)
    assert rec["protein_coding_in_all_cells"] == 5 and rec["chosen"] == 2
    genes = R.read_tsv(tmp_path / "root" / "genes.tsv")
    # G1 has the largest spread, G3 next; G2, G4, G5 are constant (variance 0)
    assert [g["gene_id"] for g in genes] == ["ENSG01", "ENSG03"]
    hdr, *body = (tmp_path / "root" / "expression.tsv").read_text().splitlines()
    assert hdr.split("\t") == ["cell", "ENSG01", "ENSG02", "ENSG03", "ENSG04", "ENSG05"]
    assert [b.split("\t")[0] for b in body] == ["cell_a", "cell_b", "cell_e"]
    assert float(body[1].split("\t")[1]) == pytest.approx(np.log1p(100), rel=1e-5)
    # deterministic: the same inputs give the same file
    first = (tmp_path / "root" / "genes.tsv").read_bytes()
    R.build_vector(rows, gt, tmp_path / "root", files, n=2)
    assert (tmp_path / "root" / "genes.tsv").read_bytes() == first


def test_check_counts_and_md5(listed, tmp_path, capsys):
    rows, _ = listed
    files, root, tracks = tmp_path / "files", tmp_path / "root", tmp_path / "tracks"
    R.write_tsv(root / "rna_cells.tsv", rows, R.COLUMNS)
    cells = R.read_tsv(root / "rna_cells.tsv")
    for it in R.file_items(cells, files):
        p = Path(it["dest"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(it["accession"].encode())
    # rewrite md5 columns to the true md5 of the written bytes
    for c in cells:
        c["md5"] = ";".join(f"{kv.split(':')[0]}:{hashlib.md5(kv.split(':')[0].encode()).hexdigest()}"
                            for kv in c["md5"].split(";"))
    R.write_tsv(root / "rna_cells.tsv", cells, R.COLUMNS)
    z = {c: np.zeros(2, np.float16) for c in SIZES}
    for c in cells:
        R.write_track(tracks / f"{c['cell']}.h5", c["cell"], z, z, {})
    (root / "expression.tsv").write_text(
        "cell\tENSG01\n" + "".join(f"{c['cell']}\t0.5\n" for c in cells))
    assert R.check(root, tracks, files_dir=files) == 0
    out = capsys.readouterr().out
    assert "cells with tracks: 3" in out and "cells with vector: 3" in out
    assert "md5 mismatches: 0" in out
    p = R.local_path(cells[0], "plus", files)
    p.write_bytes(b"corrupt")
    (tracks / "cell_b.h5").unlink()
    assert R.check(root, tracks, files_dir=files) == 1
    out = capsys.readouterr().out
    assert "md5 mismatches: 1" in out and "cells with tracks: 2" in out
    assert "cell_b: track: no track file" in out
