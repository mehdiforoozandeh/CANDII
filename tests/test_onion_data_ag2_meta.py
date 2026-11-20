"""Tests for tools/onion_data/ag2_meta.py (chunk AG0). A fake client; no `alphagenome` needed."""
from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.onion_data import ag2_meta as M  # noqa: E402


def test_import_does_not_import_alphagenome():
    code = ("import sys; sys.path.insert(0, %r); import tools.onion_data.ag2_meta; "
            "print(any(m == 'alphagenome' or m.startswith('alphagenome.') for m in sys.modules))"
            % str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --------------------------------------------------------------------------- synthetic listing

def _chip_md(cond, marks_ends, mode="observed", src="encode"):
    return pd.DataFrame([{
        "name": f"{cond}_Histone ChIP-seq-encode-{mk}-{en}-ended", "strand": ".",
        "biosample_curie": cond, "imputation_mode": mode, "biosample_type": "cell_line",
        "biosample_name": f"name {cond}", "Assay title": "Histone ChIP-seq", "data_source": src,
        "histone_mark": mk, "genetically_modified": False, "endedness": en,
        "data_pipeline": "encode"} for mk, en in marks_ends])


def _acc_md(cond, assay, title, ends, mode="observed"):
    return pd.DataFrame([{
        "name": f"{cond}_{title}-encode-{en}-ended", "strand": ".", "biosample_curie": cond,
        "imputation_mode": mode, "biosample_type": "cell_line", "biosample_name": f"name {cond}",
        "Assay title": title, "data_source": "encode", "genetically_modified": False,
        "endedness": en, "data_pipeline": "encode"} for en in ends])


CHIP_CH = [("H3K27ac", "single"), ("H3K27ac", "paired"), ("H3K4me3", "single")]


def _listing():
    chip = pd.concat([_chip_md("EFO:0002067", CHIP_CH),
                      _chip_md("UBERON:0000178_CL:0000084", [("H3K4me3", "single")])])
    dnase = _acc_md("EFO:0002067", "DNASE", "DNase-seq", ["single", "paired"])
    atac = _acc_md("CL:0000100", "ATAC", "ATAC-seq", ["paired"])
    observed = {"CHIP_HISTONE": chip, "DNASE": dnase, "ATAC": atac}
    supported = pd.DataFrame({
        "condition": ["EFO:0002067", "UBERON:0000178_CL:0000084", "CL:0000100", "EFO:0009999"],
        "biosample_type": ["cell_line"] * 4,
        "biosample_name": ["K562", "blood T cell", "motor neuron", "nothing observed"],
        "dnase": ["observed", "imputed_supported", "imputed_supported", "imputed_exploratory"],
        "atac": ["imputed_supported", "imputed_exploratory", "observed", "imputed_exploratory"],
    })
    pred = {"CHIP_HISTONE": _chip_md("EFO:0002067", CHIP_CH + [("H3K9me3", "single")]),
            "DNASE": _acc_md("EFO:0002067", "DNASE", "DNase-seq", ["paired", "single"]),
            "ATAC": _acc_md("EFO:0002067", "ATAC", "ATAC-seq", ["paired", "single"],
                            mode="imputed_supported")}
    channels = M.normalize_prediction_channels(pred, "EFO:0002067")
    return observed, supported, channels


def test_normalize_tables():
    observed, supported, channels = _listing()
    obs, chan, cond = M.normalize(observed, supported, channels)
    assert list(obs.columns) == M.OBSERVED_COLUMNS
    assert list(chan.columns) == M.CHANNEL_COLUMNS
    assert list(cond.columns) == M.CONDITION_COLUMNS
    assert len(obs) == 4 + 2 + 1
    assert (obs["n_observed_tracks"] == 1).all()
    k = obs[(obs.condition == "EFO:0002067") & (obs.output_type == "CHIP_HISTONE")]
    assert list(k["channel"]) == ["Histone ChIP-seq-encode-H3K27ac-single-ended",
                                  "Histone ChIP-seq-encode-H3K27ac-paired-ended",
                                  "Histone ChIP-seq-encode-H3K4me3-single-ended"]
    assert list(k["channel_index"]) == [0, 1, 2]
    assert list(k["target"]) == ["H3K27ac", "H3K27ac", "H3K4me3"]
    d = obs[obs.output_type == "DNASE"]
    assert dict(zip(d.endedness, d.channel_index)) == {"paired": 0, "single": 1}
    assert (d["target"] == "").all()
    assert len(chan) == 4 + 2 + 2
    c = cond.set_index("condition")
    assert c.loc["UBERON:0000178_CL:0000084", "biosample_curies"] == "UBERON:0000178;CL:0000084"
    assert c.loc["EFO:0002067", "n_observed_encode_bulk"] == 5
    assert c.loc["EFO:0002067", "has_observed_accessibility"] == 1
    assert c.loc["UBERON:0000178_CL:0000084", "has_observed_accessibility"] == 0
    assert c.loc["CL:0000100", "has_observed_accessibility"] == 1
    assert c.loc["EFO:0009999", "n_observed_encode_bulk"] == 0


def test_encode_bulk_counts_only_encode_source():
    observed, supported, channels = _listing()
    observed["CHIP_HISTONE"] = pd.concat([
        observed["CHIP_HISTONE"],
        _chip_md("EFO:0009999", [("H3K4me3", "single")], src="roadmap")])
    obs, _, cond = M.normalize(observed, supported, channels)
    assert "roadmap" in set(obs.data_source)
    assert cond.set_index("condition").loc["EFO:0009999", "n_observed_encode_bulk"] == 0


def test_normalize_rejects_inconsistent_inputs():
    observed, supported, channels = _listing()
    bad = supported.copy()
    bad.loc[bad.condition == "CL:0000100", "atac"] = "imputed_supported"
    with pytest.raises(ValueError, match="disagree"):
        M.normalize(observed, bad, channels)
    with pytest.raises(ValueError, match="not in the channel listing"):
        M.normalize(observed, supported, channels[channels.output_type != "ATAC"])
    obs2 = dict(observed)
    obs2["DNASE"] = observed["DNASE"].assign(imputation_mode="imputed_supported")
    with pytest.raises(ValueError, match="imputation modes"):
        M.normalize(obs2, supported, channels)
    with pytest.raises(ValueError):
        M.split_curies("EFO:0002067_not a curie")


# --------------------------------------------------------------------------- key rule

def _key(tmp_path, text, mode=0o600):
    p = tmp_path / "key"
    p.write_text(text)
    os.chmod(p, mode)
    return p


def test_key_file_rules(tmp_path, monkeypatch):
    assert M.read_key(_key(tmp_path, "abc123\n")) == "abc123"
    monkeypatch.setenv("AG2_KEY_FILE", str(_key(tmp_path, "zz9\n")))
    assert M.read_key() == "zz9"
    for text in ["", "\n", "a\nb\n", "a\n\n", "\nabc\n", " abc\n"]:
        with pytest.raises(SystemExit):
            M.read_key(_key(tmp_path, text))
    with pytest.raises(SystemExit, match="mode"):
        M.read_key(_key(tmp_path, "abc\n", mode=0o644))


def test_error_name_is_code_name_only():
    class Code:
        name = "RESOURCE_EXHAUSTED"

    class RpcError(Exception):
        def code(self):
            return Code()

    assert M.error_name(RpcError("project 1234567 quota")) == "RESOURCE_EXHAUSTED"
    assert M.error_name(ValueError("secret text")) == "ValueError"


# --------------------------------------------------------------------------- fake client

class FakeTrack:
    def __init__(self, md, values, resolution):
        self.metadata, self.values, self.resolution = md, values, resolution


class FakeClient:
    def __init__(self, together=True):
        self.together = together
        self.calls = []
        observed, supported, _ = _listing()
        self._obs, self._sup = observed, supported

    def observed_metadata(self, organism):
        return types.SimpleNamespace(**{M.ATTR[k]: v for k, v in self._obs.items()})

    def supported_conditions(self, organism):
        return self._sup

    def predict_interval(self, iv, *, organism, requested_outputs, condition):
        self.calls.append((iv, tuple(requested_outputs), condition))
        if not self.together and len(requested_outputs) > 1:
            raise _Rpc()
        _, _, _ = _listing()
        out = {}
        rng = np.random.default_rng(0)
        for t in requested_outputs:
            if t == "CHIP_HISTONE":
                md = _chip_md(condition, CHIP_CH + [("H3K9me3", "single")], mode="imputed_supported")
                out["chip_histone"] = FakeTrack(md, rng.gamma(2.0, 0.5, (8192, len(md))), 128)
            else:
                title = "DNase-seq" if t == "DNASE" else "ATAC-seq"
                md = _acc_md(condition, t, title, ["paired", "single"])
                out[M.ATTR[t]] = FakeTrack(md, np.full((64, 2), 0.03), 1)
        return types.SimpleNamespace(**{a: out.get(a) for a in M.ATTR.values()})


class _Rpc(Exception):
    def code(self):
        return types.SimpleNamespace(name="INVALID_ARGUMENT")


def _fake_api():
    return types.SimpleNamespace(
        version="2.0.0-fake",
        OutputType=types.SimpleNamespace(**{t: t for t in M.OUTPUT_TYPES}),
        Organism=types.SimpleNamespace(HOMO_SAPIENS="human"),
        Interval=lambda chrom, s, e: (chrom, s, e))


@pytest.mark.parametrize("together", [True, False])
def test_smoke_with_fake_client(tmp_path, together):
    client = FakeClient(together=together)
    f = M.run_smoke(client, _fake_api(), tmp_path, condition="EFO:0002067", max_calls=6)
    assert len(client.calls) <= M.MAX_SMOKE_CALLS and len(client.calls) == 6
    assert {c[2] for c in client.calls} == {"EFO:0002067"}
    text = (tmp_path / "API_SMOKE.md").read_text()
    lines = [ln for ln in text.splitlines() if ln.startswith("one call, several output types:")]
    assert lines == [f"one call, several output types: {'yes' if together else 'no'}"]
    assert "2.0.0-fake" in text and "(1183, 12)" in text
    assert "Verdict: **mean**" in text
    assert "INVALID_ARGUMENT" in text if not together else True
    ch = pd.read_csv(tmp_path / "prediction_channels.tsv", sep="\t", keep_default_na=False)
    assert len(ch) == 4 + 2 + 2
    # metadata from the smoke's channel listing, end to end with the fake client
    obs, chan, cond = M.run_metadata(client, "human", tmp_path, tmp_path / "prediction_channels.tsv",
                                     "2.0.0-fake")
    for n in ("ag2_observed.tsv", "ag2_channels.tsv", "ag2_conditions.tsv", "META_SCHEMA.md"):
        assert (tmp_path / n).exists()
    assert "The `data_source` value that means ENCODE is `encode`" in (
        tmp_path / "META_SCHEMA.md").read_text()
    back = pd.read_csv(tmp_path / "ag2_observed.tsv", sep="\t", keep_default_na=False)
    assert list(back.columns) == M.OBSERVED_COLUMNS and len(back) == len(obs)


def test_smoke_call_budget(tmp_path):
    with pytest.raises(SystemExit):
        M.run_smoke(FakeClient(), _fake_api(), tmp_path, max_calls=11)


def test_missing_accessibility_column_raises():
    observed, supported, channels = _listing()
    for col in ("dnase", "atac"):
        with pytest.raises(ValueError, match="no '%s' column" % col):
            M.normalize(observed, supported.drop(columns=[col]), channels)


SECRET = "PLANTEDKEY0123456789secret"


class _LeakyRpc(Exception):
    def code(self):
        return types.SimpleNamespace(name="PERMISSION_DENIED")


class LeakyClient(FakeClient):
    """Every API error message quotes the key, as a careless server might."""

    def __init__(self, fail):
        super().__init__(together=True)
        self.fail = fail

    def observed_metadata(self, organism):
        if self.fail == "observed":
            raise _LeakyRpc(f"denied for key {SECRET}")
        return super().observed_metadata(organism)

    def supported_conditions(self, organism):
        if self.fail == "supported":
            raise RuntimeError(f"bad key {SECRET}")
        return super().supported_conditions(organism)

    def predict_interval(self, iv, **kw):
        if self.fail == "predict":
            raise _LeakyRpc(f"quota for key {SECRET}")
        return super().predict_interval(iv, **kw)


@pytest.mark.parametrize("fail", ["create", "observed", "supported", "predict"])
@pytest.mark.parametrize("cmd", ["smoke", "metadata"])
def test_key_never_leaks(tmp_path, monkeypatch, capsys, fail, cmd):
    monkeypatch.setenv("AG2_KEY_FILE", str(_key(tmp_path, SECRET + "\n")))
    out = tmp_path / "out"
    out.mkdir()
    # metadata needs a channel listing; take it from a clean one-call smoke
    M.run_smoke(FakeClient(), _fake_api(), out, condition="EFO:0002067", max_calls=1)
    seen_keys = []

    def create(key):
        seen_keys.append(key)
        if fail == "create":
            raise RuntimeError(f"invalid key {key}")
        return LeakyClient(fail)

    api = _fake_api()
    api.create = create
    errors = []
    try:
        rc = M.main([cmd, "--out", str(out)], api=api)
    except BaseException as exc:  # noqa: BLE001
        rc = None
        errors.append(repr(exc) + str(exc))
    assert seen_keys == [SECRET]  # the tool read the key and passed it to the client only
    cap = capsys.readouterr()
    texts = [cap.out, cap.err, *errors] + [p.read_text() for p in out.iterdir()]
    assert all(SECRET not in t for t in texts)
    expect_fail = fail in ("create", "observed") or (fail == "supported" and cmd == "metadata")
    if expect_fail:
        assert rc == 1 and "failed:" in cap.err
    else:
        assert rc == 0, errors
    if fail == "predict" and cmd == "smoke":
        assert "PERMISSION_DENIED" in (out / "API_SMOKE.md").read_text()


def test_chip_verdict():
    assert M.chip_verdict(1.1) == "mean"
    assert M.chip_verdict(128.0) == "sum"
    assert M.chip_verdict(16.0) == "undecided"
