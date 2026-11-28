"""t396 C12a: the 1 bp dedup tool on synthetic tagAlign lines."""
import gzip
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from t396 import dedup_1bp as d  # noqa: E402

READ_LEN = 36


def _reads(rng, n_pos, mean_dup, chrom="chr1", span=200_000):
    """Positions (pos5, strand) with a geometric number of duplicate reads each."""
    pos = rng.choice(span, size=n_pos, replace=False) + 1000
    strand = rng.integers(0, 2, n_pos)
    k = rng.geometric(1.0 / mean_dup, n_pos)
    return [(chrom, int(p), int(s)) for p, s, kk in zip(pos, strand, k) for _ in range(kk)]


def _write(path, reads):
    with gzip.open(path, "wt") as f:
        for c, p, s in reads:
            start, end = (p, p + READ_LEN) if s == 0 else (p - READ_LEN + 1, p + 1)
            f.write(f"{c}\t{start}\t{end}\tN\t1000\t{'+-'[s]}\n")
    return str(path)


def _distinct(reads):
    return sorted(set(reads))


def _run_unit(tmp, exp, label, n_units, off, on):
    out = str(tmp / f"{exp}_{label}.json")
    d.main(["unit", "--off", *off, "--on", *on, "--experiment", exp, "--label", label,
            "--n-units", str(n_units), "--out", out])
    return out


def _experiment(tmp, rng, exp, pooled_dedup):
    """Two replicates that share positions; dedup per replicate or over the pooled union."""
    shared = _reads(rng, 300, 2.0)
    r1 = shared + _reads(rng, 500, 1.5)
    r2 = list(shared) + _reads(rng, 500, 1.5)
    files, outs = {}, []
    for i, r in enumerate((r1, r2), 1):
        files[i] = {"off": [_write(tmp / f"{exp}_br{i}_off_pr1.gz", r[::2]),
                            _write(tmp / f"{exp}_br{i}_off_pr2.gz", r[1::2])]}
    if pooled_dedup:
        seen, on = set(), {1: [], 2: []}
        for i, r in ((1, r1), (2, r2)):
            for x in _distinct(r):
                if x not in seen:
                    seen.add(x)
                    on[i].append(x)
    else:
        on = {1: _distinct(r1), 2: _distinct(r2)}
    for i in (1, 2):
        files[i]["on"] = [_write(tmp / f"{exp}_br{i}_on.gz", on[i])]
        outs.append(_run_unit(tmp, exp, f"br{i}", 1, files[i]["off"], files[i]["on"]))
    outs.append(_run_unit(tmp, exp, "pooled", 2, files[1]["off"] + files[2]["off"],
                          files[1]["on"] + files[2]["on"]))
    return outs


def test_keys_use_5prime_end():
    idx = d.ChromIndex()
    k = d.keys_from_columns(idx.code(np.array(["chr1", "chr1"])), [100, 100], [136, 136],
                            np.array([False, True]))
    assert (k >> 1 & ((1 << 32) - 1)).tolist() == [100, 135]
    assert (k & 1).tolist() == [0, 1]


@pytest.mark.parametrize("pooled_dedup,expected", [(False, "replicate"), (True, "pooled")])
def test_rule(tmp_path, pooled_dedup, expected):
    rng = np.random.default_rng(1)
    outs = []
    for e in ("E1", "E2"):
        outs += _experiment(tmp_path, rng, e, pooled_dedup)
    rule = tmp_path / "UNIT_RULE.json"
    d.main(["rule", "--units", *outs, "--out", str(rule)])
    r = json.load(open(rule))
    assert r["dedup_unit"] == expected
    assert r["on_equals_distinct_off"] is True
    assert r["max_share_absent"] == 0.0
    assert r["n_experiments"] == 2


def test_unit_shares(tmp_path):
    rng = np.random.default_rng(2)
    off = _reads(rng, 400, 3.0)
    on = _distinct(off)[:-40] + [("chr2", 5, 0)] * 10     # 40 lost, 1 new position repeated
    u = d.main(["unit", "--off", _write(tmp_path / "off.gz", off), "--on",
                _write(tmp_path / "on.gz", on), "--experiment", "E", "--label", "br1",
                "--out", str(tmp_path / "u.json")])
    assert u["n_off_distinct"] == 400
    assert u["n_off_absent_from_on"] == 40
    assert u["n_on_absent_from_off"] == 1
    assert u["share_on_repeated_lines"] == pytest.approx(9 / len(on))
    r = d.decide([u])
    assert r["dedup_unit"] == "other" and r["on_equals_distinct_off"] is False


def test_dm_shape_recovered():
    rng = np.random.default_rng(3)
    a, K = 0.3, d.RES
    hist_c, hist_x = {}, {}
    for _ in range(20000):
        c = int(rng.integers(2, 30))
        x = rng.multinomial(c, rng.dirichlet([a] * K))
        hist_c[c] = hist_c.get(c, 0) + 1
        for v in x[x > 0]:
            hist_x[int(v)] = hist_x.get(int(v), 0) + 1
    assert d.fit_dm_shape(hist_c, hist_x, K) == pytest.approx(a, rel=0.1)


def test_shape(tmp_path):
    rng = np.random.default_rng(4)
    off = _reads(rng, 3000, 2.0, span=20_000)
    on = _distinct(off)
    s = d.main(["shape", "--off", _write(tmp_path / "off.gz", off), "--on",
                _write(tmp_path / "on.gz", on), "--experiment", "E", "--label", "br1",
                "--out", str(tmp_path / "s.json")])
    assert sum(s["hist_off_reads_per_position"].values()) == 3000
    assert sum(k * n for k, n in s["hist_off_reads_per_position"].items()) == len(off)
    assert s["top1pct"]["sum_on"] == s["top1pct"]["sum_distinct_off"]
    assert s["n_on_reads_outside_off_bins"] == 0
    assert 1e-3 <= s["dirichlet_a_1bp"] <= 1e3


def test_shape_with_caps(tmp_path):
    pytest.importorskip("t396.caps")
    rng = np.random.default_rng(5)
    off = _reads(rng, 3000, 2.0, span=20_000)
    s = d.main(["shape", "--off", _write(tmp_path / "off.gz", off), "--on",
                _write(tmp_path / "on.gz", _distinct(off)), "--experiment", "E",
                "--label", "br1", "--out", str(tmp_path / "s.json")])
    assert s["K_25bp"] == 25.0
    t = s["top1pct"]["dm_distinct_a_25bp"]
    assert abs(t["rel_err_vs_distinct_off"]) < 0.2
    assert t["deviance_distinct_off"] == pytest.approx(t["deviance_on"])
