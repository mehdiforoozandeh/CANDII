"""t81 — Avocado under a `regions` regime (D32, `plan/BENCHMARK_DESIGN.md` §3.1).

`configs/regime.eic_pilot.json` restricts the training scope to the 44 ENCODE Pilot Regions lifted
to hg38. Until this landed, `competitors/avocado/` ignored the `regions` key: it binned whole
chromosomes and trained on all of them, which would have made its `eic.pilot` row a claim about a
scope it never used. The launcher refused the regime outright rather than run that.

The claim these tests defend is a containment claim, and it is checked twice: once as arithmetic
against the number §3.1 pins (1,023,489 bins over the shipped BED), and once behaviourally — after
a real training run, the genomic factors of every position OUTSIDE the BED are bit-identical to
their initialisation, because no gradient ever reached them.

Nothing is mocked. `make_store` writes an actual `CANDI_STORE`, `bin_store.py` reads it and
`train.py` trains on what it wrote.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_AVO = Path(__file__).resolve().parents[1] / "competitors" / "avocado"
sys.path.insert(0, str(_AVO / "vendor"))
sys.path.insert(0, str(_AVO))

import bin_store                                                            # noqa: E402
import index                                                                # noqa: E402
import train as avo_train                                                   # noqa: E402
from avocado import Avocado, G5K_STRIDE                                     # noqa: E402
from index import RegionScopeError, read_layout, region_layout, region_slots  # noqa: E402

from tests.test_store_reader import make_store                              # noqa: E402
from tests.test_store_regime import regime_dict                             # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = 25

#: `V_*` carries an assay its `T_*` prompt lacks — the only layout that poses imputation.
TRACKS = {
    "T_aa": ("ATAC-seq", "DNase-seq"),
    "T_bb": ("ATAC-seq", "DNase-seq"),
    "V_aa": ("ATAC-seq", "H3K4me3"),
    "V_bb": ("ATAC-seq", "H3K4me3"),
}

#: Two regions on chr1 (2,000 bins in the synthetic store), with boundaries deliberately OFF the
#: 25 bp grid so containment cannot be mistaken for a division.
BED = "chr1\t1013\t9987\tENr001\nchr1\t20001\t26999\tENr002\n"


@pytest.fixture(scope="module")
def scope(tmp_path_factory):
    """A store, a BED and a regime that declares it."""
    d = tmp_path_factory.mktemp("avoregions")
    store = make_store(d / "s", tracks=TRACKS)
    bed = d / "pilot.bed"
    bed.write_text(BED, encoding="utf-8")
    obj = regime_dict(store, biosamples={"train": ["T_aa", "T_bb"], "eval": ["V_aa", "V_bb"]},
                      kinds=["counts", "peaks", "pval"],
                      eval_pairs=[["T_aa", "V_aa"], ["T_bb", "V_bb"]],
                      regions={"bed": str(bed), "sha256": hashlib.sha256(bed.read_bytes()).hexdigest(),
                               "policy": "contain"})
    rp = d / "regime.json"
    rp.write_text(json.dumps(obj), encoding="utf-8")
    return d, rp, obj


# ---------------------------------------------------------------------------
# the rule: containment, at Avocado's own unit
# ---------------------------------------------------------------------------

def test_a_bin_counts_only_when_it_lies_wholly_inside_a_region(scope):
    """D32 is containment.

    `[1013, 9987)` contains bins 41..398: bin 40 starts at 1000, before the region, and bin 399 ends
    at 10000, past it. `[20001, 26999)` contains 801..1078 by the same reasoning. Both boundaries sit
    off the 25 bp grid on purpose — a division would have said 40 and 800.
    """
    _, rp, _ = scope
    spans, _ = region_layout(rp, ["chr1"])
    assert [(c, a, b) for c, a, b, _ in spans] == [("chr1", 41, 399), ("chr1", 801, 1079)]
    assert sum(b - a for _, a, b, _ in spans) == 358 + 278


def test_the_shipped_pilot_bed_plans_the_training_bins_section_3_1_pins():
    """The number the design doc quotes, recomputed from the BED the regime actually names."""
    rp = REPO / "configs" / "regime.eic_pilot.json"
    obj = json.loads(rp.read_text(encoding="utf-8"))
    spans, _ = region_layout(rp, obj["train_chroms"])
    assert len(spans) == 40, "Rule 2 cuts the four regions on the eval chromosomes"
    assert sum(b - a for _, a, b, _ in spans) == 1_023_489


def test_the_coarse_factor_grids_stay_anchored_at_chromosome_bin_zero(scope):
    """§3.1 keeps the grid anchored at bin 0 and never re-anchors it per region.

    Avocado's 250 bp and 5 kbp factors are `pos // 10` and `pos // 200`, so the packing is only
    faithful if every region's slot offset is a whole number of 5 kbp cells away from its absolute
    first bin — then a coarse cell covers exactly the absolute bins it would have covered on a
    whole-chromosome fit.
    """
    _, rp, _ = scope
    spans, n_slots = region_layout(rp, ["chr1"])
    for _, first, _, slot0 in spans:
        assert (slot0 - first) % G5K_STRIDE == 0
    cells = [(s0 // G5K_STRIDE, (s0 + b - a - 1) // G5K_STRIDE) for _, a, b, s0 in spans]
    assert all(cells[i][1] < cells[i + 1][0] for i in range(len(cells) - 1)), \
        "two regions share a 5 kbp genomic factor"
    assert n_slots >= sum(b - a for _, a, b, _ in spans)


def test_a_bed_that_misses_every_train_chromosome_is_refused(scope, tmp_path):
    """Rule 2 cuts regions by the chromosome list, so an empty intersection is a broken regime."""
    _, rp, _ = scope
    with pytest.raises(RegionScopeError):
        region_layout(rp, ["chr2"])


# ---------------------------------------------------------------------------
# the matrix, and what the fit is allowed to touch
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def binned(scope):
    d, rp, _ = scope
    out = d / "binned"
    assert bin_store.main(["--regime", str(rp), "--out", str(out), "--regions"]) == 0
    return out


def test_the_binned_matrix_carries_the_signal_of_the_bins_inside_the_bed(scope, binned):
    """Slot i must hold the absolute bin the layout says it holds — checked against the store."""
    _, rp, obj = scope
    from candi.store.reader import CorpusStore

    Y = np.load(binned / "regions.npy")
    spans, n_slots = read_layout(binned / "regions_layout.csv")
    rows = index.read_tracks(binned / "tracks.csv")
    assert Y.shape == (n_slots, len(rows))

    with CorpusStore(obj["store"]) as corpus:
        for col, bios, assay, _, _ in rows:
            for chrom, first, end, slot0 in spans:
                truth = corpus[bios].pval(chrom, first, end, assays=[assay])[:, 0]
                got = Y[slot0:slot0 + (end - first), col]
                assert np.array_equal(got, truth), f"{bios}/{assay} on {chrom}:{first}-{end}"


def test_training_moves_no_genomic_factor_outside_the_bed(scope, binned, tmp_path):
    """The behavioural half of the containment claim.

    An alignment slot is a position of the compact axis that no region covers. If the fit ever drew
    one, its 25 bp factor would have taken a gradient step. Compared against a model built with the
    same seed, so the reference is the exact initialisation this run started from.
    """
    _, rp, _ = scope
    out = tmp_path / "ckpt" / "shared_regions.pt"
    assert avo_train.main([
        "--regime", str(rp), "--chrom", "regions", "--mode", "shared",
        "--data-root", str(binned), "--out", str(out),
        "--positions", str(binned / "regions_layout.csv"),
        "--epochs", "2", "--batch-positions", "32", "--seed", "0"]) == 0

    ck = torch.load(out, map_location="cpu", weights_only=False)
    spans, n_slots = read_layout(binned / "regions_layout.csv")
    trained = set(region_slots(spans).tolist())
    idle = sorted(set(range(n_slots)) - trained)
    assert idle, "this fixture is pointless without alignment slots to check"

    torch.manual_seed(0)
    fresh = Avocado(len(ck["cells"]), len(ck["assays"]), n_slots)
    g25, ref = ck["model"]["g25.weight"], fresh.g25.weight.detach()
    assert torch.equal(g25[idle], ref[idle]), "a position outside the BED took a gradient step"
    assert not torch.equal(g25[sorted(trained)], ref[sorted(trained)]), \
        "no position inside the BED moved — the fit trained on nothing"
    assert ck["bed_scope"]["trained_slots"] == len(trained)
