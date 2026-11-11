#!/usr/bin/env python3
"""Frozen sequence-model embeddings on the 25 bp grid, for the DNA source arms (`dna_flashzoi`, `dna_gnlm`).

Runs in the separate DNA venv (`requirements-dnaemb.txt`, built by `slurm/minicandi/dna_embed.sh`), never in the main
kit venv. It imports only numpy and h5py at module level; torch, borzoi_pytorch and transformers load only when a real
model is built. It imports nothing from `candi` or `minicandi`, so it runs in that venv as is.

    dna_embed.py precompute --source flashzoi|ntv2 --genome $O/genome --out <dir> --chroms chr21 [--device cuda]
    dna_embed.py project    --dir <dir> --fit-chroms chr1 --dim 128 [--chroms ...]
    dna_embed.py verify     --dir <dir> [--chroms ...] [--genome <genome dir>]
    dna_embed.py folds      --out <prefix> [--src <url or path of sequences_human.bed.gz>]

Output of `precompute`: `<dir>/<chrom>.h5` (dataset `emb`, float16 `[n_bins, d]`, chunks of 4096 bins; n_bins is the
length of the genome layer's `mask.h5[chrom]`, i.e. floor(chrom length / 25)) and `<dir>/meta.json`. `project` fits a
PCA on random valid bins (mask.h5 = 1) of the fit chromosomes (no label is read), writes `<dir>/projection.npz` and
replaces each `<chrom>.h5` by its projection in place (the unprojected file is the deleted intermediate).

Grid rule (pure function `native_index`): native grids do not nest in 25 bp (Flashzoi 32 bp, the 6-mer tokenizer of
Nucleotide Transformer v2 6 bp). The 25 bp bin b covers [25 b, 25 b + 25) bp; it takes the embedding of the native bin
that holds its centre 25 b + 12.5, i.e. native index floor((25 b + 12.5) / native_bp).

Tiling: a model sees `input_bp` of sequence and its embedding is kept for the central `out_bp` only (`crop_bp` on each
side is context). Window k keeps [k out_bp, (k + 1) out_bp); positions outside the chromosome are N (code 4).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import h5py
import numpy as np

RES = 25                      # the 25 bp grid (tools/minicandi/contract.py RES; a test pins the two equal)
CHUNK_BINS = 4096
N_CODE = 4                    # dna.h5 base codes: A=0 C=1 G=2 T=3 N=4
CANONICAL = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
DEFAULT_WEIGHTS_DIR = "/project/def-maxwl/mforooz/ONION/dna_models"
BORZOI_BED_URL = "https://raw.githubusercontent.com/calico/borzoi/main/data/sequences_human.bed.gz"
# md5 of Max's copy of the same file (Fir /project/def-maxwl/maxwl/epi_imputation/s02/encoder/folds/, read 2026-11-11):
BORZOI_BED_MD5_REF = "a32f2b21391403252e864e2f3543639a"
BORZOI_TEST_FOLD = "fold3"    # the released replicates (Flashzoi too): test fold3, validation fold4 (Borzoi README)


# ----------------------------------------------------------------------------------------------- grid rule
def native_index(n_bins: int, native_bp: int, res: int = RES) -> np.ndarray:
    """Native bin index of each 25 bp bin: the native bin [i native_bp, (i + 1) native_bp) that holds the bin's centre.

    Integer form of floor((res b + res / 2) / native_bp), exact for any res and native_bp."""
    b = np.arange(int(n_bins), dtype=np.int64)
    return (2 * res * b + res) // (2 * int(native_bp))


# ----------------------------------------------------------------------------------------------- sources
@dataclass
class Source:
    """A frozen model: `embed(codes uint8 [B, input_bp]) -> float32 [B, out_bp / native_bp, d]` (central part only)."""
    name: str
    native_bp: int
    input_bp: int
    out_bp: int
    d: int
    embed: Callable[[np.ndarray], np.ndarray]
    meta: dict = field(default_factory=dict)

    @property
    def crop_bp(self) -> int:
        return (self.input_bp - self.out_bp) // 2

    def check(self) -> None:
        if self.out_bp % self.native_bp or (self.input_bp - self.out_bp) % 2 or self.out_bp > self.input_bp:
            raise ValueError(f"{self.name}: bad geometry input {self.input_bp}, out {self.out_bp}, "
                             f"native {self.native_bp}")


# Pinned model identities: checked 2026-11-11 against the Hugging Face API (sha = repository revision; sha256 = the LFS
# sha256 of the weights file). Flashzoi's sha256 also equals Max's Fir copy of the same revision.
MODELS = {
    "flashzoi": dict(repo="johahi/flashzoi-replicate-0", revision="72040fb525d4b9abab21ead1c53b76b6bd8b6d67",
                     weights_file="model.safetensors",
                     sha256="d0f17287f57da22af70a495d27e745b7d9908d2f3717214e7d0caf3dfe8c4d95",
                     license="MIT (model card); borzoi-pytorch code Apache-2.0",
                     trained_on_encode_tracks=True),
    "ntv2": dict(repo="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
                 revision="06615c1660c892fc199840c18123f8385b3542a8", weights_file="model.safetensors",
                 sha256="7a00ffaaf8f3ca627741a61e568d7e550b5c79aada70c76c8cb861192fbe7c59",
                 license="CC BY-NC-SA 4.0 (model card)", trained_on_encode_tracks=False),
}

# Flashzoi: 524,288 bp in, the pre-head embedding (final_joined_convs, 1920 channels) over the central 6,144 x 32 bp
# out, as Max's sprint 02 cache (scripts/encoder/common.py embed32).
FLASHZOI_GEOM = dict(native_bp=32, input_bp=524_288, out_bp=6_144 * 32, d=1920)
# NT v2: 2,048 tokens at most including <cls>; 2,046 6-mers in (12,276 bp), the central 1,364 kept (crop 341 tokens).
NT_TOKENS_IN, NT_TOKENS_KEEP = 2046, 1364
NTV2_GEOM = dict(native_bp=6, input_bp=NT_TOKENS_IN * 6, out_bp=NT_TOKENS_KEEP * 6, d=1024)


def sha256_file(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights(source: str, weights: Optional[str], weights_dir: str) -> tuple[Path, str]:
    """Local model directory and its weights sha256; refuses a sha256 that differs from the pinned one."""
    m = MODELS[source]
    if weights:
        local = Path(weights)
    else:
        from huggingface_hub import snapshot_download
        local = Path(snapshot_download(m["repo"], revision=m["revision"], cache_dir=weights_dir))
    sha = sha256_file(local / m["weights_file"])
    if sha != m["sha256"]:
        raise SystemExit(f"{local / m['weights_file']}: sha256 {sha} is not the pinned {m['sha256']}")
    return local, sha


def nt_token_ids(codes: np.ndarray, kmer_to_id: dict, unk_id: int, cls_id: int) -> np.ndarray:
    """codes uint8 [B, 6 T] -> int64 [B, 1 + T]: <cls>, then one id per non-overlapping 6-mer.

    Position-exact by construction: a 6-mer that holds an N (no such token exists) becomes <unk>, so token t always
    covers bases [6 t, 6 t + 6). The model's own tokenizer would split it into single bases and shift the grid."""
    B, Lbp = codes.shape
    if Lbp % 6:
        raise ValueError(f"{Lbp} bp is not a whole number of 6-mers")
    k = codes.reshape(B, Lbp // 6, 6).astype(np.int64)
    has_n = (k >= N_CODE).any(-1)
    key = np.zeros(k.shape[:2], np.int64)
    for j in range(6):
        key = key * 4 + np.minimum(k[..., j], 3)
    lut = np.full(4 ** 6, unk_id, np.int64)
    letters = "ACGT"
    for kmer, tid in kmer_to_id.items():
        if len(kmer) == 6 and set(kmer) <= set(letters):
            v = 0
            for ch in kmer:
                v = v * 4 + letters.index(ch)
            lut[v] = tid
    ids = np.where(has_n, unk_id, lut[key])
    return np.concatenate([np.full((B, 1), cls_id, np.int64), ids], 1)


def build_source(source: str, device: str, weights: Optional[str], weights_dir: str) -> Source:
    import torch
    m = MODELS[source]
    local, sha = fetch_weights(source, weights, weights_dir)
    dev = torch.device(device)
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if dev.type == "cuda" else None
    meta = dict(model_id=m["repo"], revision=m["revision"], weights_file=m["weights_file"], weights_sha256=sha,
                weights_dir=str(local), license=m["license"], trained_on_encode_tracks=m["trained_on_encode_tracks"],
                device=device, autocast="bfloat16" if amp else "none", torch=torch.__version__)
    if source == "flashzoi":
        import borzoi_pytorch
        from borzoi_pytorch import Borzoi
        model = Borzoi.from_pretrained(str(local)).eval().to(dev)
        meta.update(borzoi_pytorch=getattr(borzoi_pytorch, "__version__", "unknown"),
                    layer="final_joined_convs(get_embs_after_crop(onehot ACGT; N = all-zero column))",
                    flashed=bool(getattr(model, "flashed", False)))

        def embed(codes: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                x = torch.from_numpy(codes).to(dev).long().clamp(max=N_CODE)
                oh = torch.nn.functional.one_hot(x, 5)[..., :4].permute(0, 2, 1).float().contiguous()
                with amp() if amp else _null():
                    z = model.final_joined_convs(model.get_embs_after_crop(oh))     # (B, 1920, 6144)
                return z.float().transpose(1, 2).cpu().numpy()
        return Source("flashzoi", embed=embed, meta=meta, **FLASHZOI_GEOM)

    import transformers
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(local), trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(str(local), trust_remote_code=True).eval().to(dev)
    vocab = tok.get_vocab()
    unk, cls = int(tok.unk_token_id), int(tok.cls_token_id)
    meta.update(transformers=transformers.__version__, layer="hidden_states[-1] (last layer), <cls> dropped",
                tokens_in=NT_TOKENS_IN, tokens_kept=NT_TOKENS_KEEP, n_kmer="6-mer holding an N -> <unk>")
    c0 = (NT_TOKENS_IN - NT_TOKENS_KEEP) // 2

    def embed(codes: np.ndarray) -> np.ndarray:
        ids = torch.from_numpy(nt_token_ids(codes, vocab, unk, cls)).to(dev)
        with torch.no_grad():
            with amp() if amp else _null():
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
            h = out.hidden_states[-1][:, 1:]                                         # (B, 2046, 1024)
            return h[:, c0:c0 + NT_TOKENS_KEEP].float().cpu().numpy()
    return Source("ntv2", embed=embed, meta=meta, **NTV2_GEOM)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ----------------------------------------------------------------------------------------------- genome
def genome_bins(genome: Path, chrom: str) -> int:
    with h5py.File(Path(genome) / "mask.h5", "r") as f:
        return int(f[chrom].shape[0])


def genome_chroms(genome: Path) -> list[str]:
    with h5py.File(Path(genome) / "mask.h5", "r") as f:
        have = set(f.keys())
    return [c for c in CANONICAL if c in have]


def padded(codes: np.ndarray, start: int, length: int) -> np.ndarray:
    """codes[start:start + length], N outside the chromosome."""
    out = np.full(length, N_CODE, np.uint8)
    s, e = max(start, 0), min(start + length, codes.shape[0])
    if e > s:
        out[s - start:e - start] = codes[s:e]
    return out


# ----------------------------------------------------------------------------------------------- meta
def write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_meta(d: Path) -> dict:
    p = Path(d) / "meta.json"
    return json.loads(p.read_text()) if p.is_file() else {}


# ----------------------------------------------------------------------------------------------- precompute
def precompute_chrom(src: Source, genome: Path, chrom: str, out: Path, batch: int = 4,
                     log: Callable[[str], None] = print) -> Path:
    """Write `<out>/<chrom>.h5`; skips a chromosome whose file exists (a crash leaves only `<chrom>.h5.tmp`)."""
    src.check()
    final = Path(out) / f"{chrom}.h5"
    if final.is_file():
        log(f"[dna_embed] {chrom}: {final} exists, skipped")
        return final
    n_bins = genome_bins(genome, chrom)
    with h5py.File(Path(genome) / "dna.h5", "r") as f:
        codes = np.asarray(f[chrom][:], np.uint8)
    idx = native_index(n_bins, src.native_bp)
    per_win = src.out_bp // src.native_bp
    n_native = int(idx[-1]) + 1 if n_bins else 0
    n_win = -(-n_native // per_win)
    tmp = final.with_name(final.name + ".tmp")
    with h5py.File(tmp, "w") as h:
        ds = h.create_dataset("emb", shape=(n_bins, src.d), dtype="float16",
                              chunks=(max(1, min(CHUNK_BINS, n_bins)), src.d))
        h.attrs.update(source=src.name, chrom=chrom, native_bp=src.native_bp, res=RES, projected_dim=0)
        for w0 in range(0, n_win, batch):
            ws = range(w0, min(w0 + batch, n_win))
            x = np.stack([padded(codes, w * src.out_bp - src.crop_bp, src.input_bp) for w in ws])
            z = np.asarray(src.embed(x), np.float32)
            if z.shape != (len(ws), per_win, src.d):
                raise ValueError(f"{src.name}: embed gave {z.shape}, expected {(len(ws), per_win, src.d)}")
            z = z.reshape(-1, src.d)
            n0, n1 = w0 * per_win, (w0 + len(ws)) * per_win
            b0, b1 = np.searchsorted(idx, n0, "left"), np.searchsorted(idx, n1, "left")
            if b1 > b0:
                ds[b0:b1] = z[idx[b0:b1] - n0].astype(np.float16)
            if w0 % (batch * 500) == 0:
                log(f"[dna_embed] {chrom}: window {w0}/{n_win} {_dt.datetime.now().isoformat(timespec='seconds')}")
    os.replace(tmp, final)
    log(f"[dna_embed] {chrom}: {n_bins} bins x {src.d} -> {final}")
    return final


def precompute(src: Source, genome: Path, out: Path, chroms: Sequence[str], batch: int = 4,
               log: Callable[[str], None] = print) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = read_meta(out)
    if meta and meta.get("source") != src.name:
        raise SystemExit(f"{out}/meta.json holds source {meta.get('source')!r}, not {src.name!r}")
    meta.update(source=src.name, native_bin_bp=src.native_bp, receptive_field_bp=src.input_bp,
                kept_span_bp=src.out_bp, d=src.d, res=RES, genome=str(Path(genome).resolve()),
                grid_rule="25 bp bin b takes native bin floor((25 b + 12.5) / native_bp)",
                dtype="float16", chunk_bins=CHUNK_BINS, date=_dt.date.today().isoformat(), **src.meta)
    meta.setdefault("projection", None)
    write_json_atomic(out / "meta.json", meta)
    for c in chroms:
        precompute_chrom(src, genome, c, out, batch=batch, log=log)


# ----------------------------------------------------------------------------------------------- project (PCA)
def _blocks(n: int, step: int = CHUNK_BINS * 16):
    for s in range(0, n, step):
        yield s, min(s + step, n)


def fit_pca(d: Path, genome: Path, fit_chroms: Sequence[str], dim: int, n_fit: int, seed: int) -> dict:
    """PCA of n_fit random valid bins (mask = 1) of the fit chromosomes: streamed mean and covariance in float64."""
    rng = np.random.default_rng(seed)
    valid = {}
    for c in fit_chroms:
        with h5py.File(Path(genome) / "mask.h5", "r") as f:
            valid[c] = np.flatnonzero(np.asarray(f[c][:]) > 0)
    tot = sum(v.size for v in valid.values())
    take = min(int(n_fit), tot)
    pick = np.sort(rng.choice(tot, size=take, replace=False))
    s1 = s2 = None
    n = 0
    off = 0
    for c in fit_chroms:
        rows = valid[c][pick[(pick >= off) & (pick < off + valid[c].size)] - off]
        off += valid[c].size
        with h5py.File(Path(d) / f"{c}.h5", "r") as h:
            ds = h["emb"]
            if int(h.attrs.get("projected_dim", 0)):
                raise SystemExit(f"{d}/{c}.h5 is already projected and {d}/projection.npz is missing: refit is "
                                 f"impossible; rerun precompute for {c}")
            for s, e in _blocks(ds.shape[0]):
                r = rows[(rows >= s) & (rows < e)]
                if not r.size:
                    continue
                x = np.asarray(ds[s:e], np.float64)[r - s]
                s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
                s2 = x.T @ x if s2 is None else s2 + x.T @ x
                n += r.size
    mean = s1 / n
    cov = (s2 - n * np.outer(mean, mean)) / max(n - 1, 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][:dim]
    comp = v[:, order].T                                           # (dim, d)
    sign = np.sign(comp[np.arange(dim), np.abs(comp).argmax(1)])
    comp = comp * np.where(sign == 0, 1, sign)[:, None]
    return dict(mean=mean.astype(np.float32), components=comp.astype(np.float32),
                eigenvalues=w[order].astype(np.float64), explained=float(w[order].sum() / max(w.sum(), 1e-300)),
                n_fit=int(n), seed=int(seed), fit_chroms=list(fit_chroms))


def project(d: Path, genome: Optional[Path], fit_chroms: Sequence[str], dim: int, chroms: Optional[Sequence[str]],
            n_fit: int = 1_000_000, seed: int = 0, log: Callable[[str], None] = print) -> None:
    d = Path(d)
    meta = read_meta(d)
    genome = Path(genome or meta["genome"])
    pfile = d / "projection.npz"
    if pfile.is_file():
        p = dict(np.load(pfile, allow_pickle=False))
        if p["components"].shape[0] != dim:
            raise SystemExit(f"{pfile} has dim {p['components'].shape[0]}, not {dim}")
        log(f"[dna_embed] reusing {pfile}")
    else:
        p = fit_pca(d, genome, fit_chroms, dim, n_fit, seed)
        np.savez(d / ".projection.tmp.npz", **{k: np.asarray(v) for k, v in p.items()})
        os.replace(d / ".projection.tmp.npz", pfile)
        log(f"[dna_embed] PCA {dim} of {p['mean'].shape[0]} on {p['n_fit']} bins: explained {p['explained']:.4f}")
    psha = sha256_file(pfile)
    mean, comp = np.asarray(p["mean"], np.float32), np.asarray(p["components"], np.float32)
    meta["projection"] = dict(method="pca", dim=int(dim), d_in=int(comp.shape[1]), file="projection.npz",
                              sha256=psha, fit_chroms=[str(c) for c in np.atleast_1d(p["fit_chroms"])],
                              n_fit=int(p["n_fit"]), seed=int(p["seed"]), explained=float(p["explained"]),
                              sample="uniform over mask.h5 = 1 bins of the fit chromosomes; no label read")
    meta["d"] = int(dim)
    write_json_atomic(d / "meta.json", meta)
    for c in (chroms or sorted_chroms(d)):
        f = d / f"{c}.h5"
        with h5py.File(f, "r") as h:
            if int(h.attrs.get("projected_dim", 0)) == dim and h.attrs.get("projection_sha256", "") == psha:
                log(f"[dna_embed] {c}: already projected, skipped")
                continue
            if int(h.attrs.get("projected_dim", 0)):
                raise SystemExit(f"{f}: projected with another projection")
            tmp = f.with_name(f.name + ".proj.tmp")
            src = h["emb"]
            with h5py.File(tmp, "w") as o:
                ds = o.create_dataset("emb", shape=(src.shape[0], dim), dtype="float16",
                                      chunks=(max(1, min(CHUNK_BINS, src.shape[0])), dim))
                o.attrs.update(dict(h.attrs))
                o.attrs.update(projected_dim=int(dim), projection_sha256=psha, d_in=int(src.shape[1]))
                for s, e in _blocks(src.shape[0]):
                    ds[s:e] = ((np.asarray(src[s:e], np.float32) - mean) @ comp.T).astype(np.float16)
        os.replace(tmp, f)
        log(f"[dna_embed] {c}: projected to {dim}")


# ----------------------------------------------------------------------------------------------- verify
def sorted_chroms(d: Path) -> list[str]:
    have = {p.name[:-3] for p in Path(d).glob("chr*.h5")}
    order = {c: i for i, c in enumerate(CANONICAL)}
    return sorted(have, key=lambda c: (order.get(c, 99), c))


def verify(d: Path, chroms: Optional[Sequence[str]] = None, genome: Optional[Path] = None) -> tuple[int, bool, int]:
    d = Path(d)
    meta = read_meta(d)
    genome = Path(genome or meta["genome"])
    chroms = list(chroms or sorted_chroms(d))
    ok, bad = True, 0
    for c in chroms:
        f = d / f"{c}.h5"
        if not f.is_file():
            print(f"[dna_embed] {c}: {f} missing", file=sys.stderr)
            ok = False
            continue
        with h5py.File(f, "r") as h:
            ds = h["emb"]
            nb = genome_bins(genome, c)
            if ds.shape[0] != nb or ds.dtype != np.float16 or ds.ndim != 2:
                print(f"[dna_embed] {c}: {ds.shape} {ds.dtype}, genome layer has {nb} bins", file=sys.stderr)
                ok = False
            for s, e in _blocks(ds.shape[0]):
                bad += int((~np.isfinite(np.asarray(ds[s:e]))).sum())
    print(f"chromosomes: {len(chroms)}, bins match genome layer: {ok}, NaN: {bad}")
    return len(chroms), ok, bad


# ----------------------------------------------------------------------------------------------- folds
def read_source_bytes(src: str) -> bytes:
    if "://" in src:
        import urllib.request
        with urllib.request.urlopen(src, timeout=120) as r:
            return r.read()
    return Path(src).read_bytes()


def folds(out: str, src: str = BORZOI_BED_URL) -> dict:
    """Borzoi training and test loci (196,608 bp target windows) as two BED files, for the leakage square.

    test = fold3; train = every other fold, validation fold4 included (Borzoi saw it for early stopping)."""
    raw = read_source_bytes(src)
    md5 = hashlib.md5(raw).hexdigest()
    text = gzip.decompress(raw).decode() if raw[:2] == b"\x1f\x8b" else raw.decode()
    rows = [ln.split("\t")[:4] for ln in text.splitlines() if ln.strip() and not ln.startswith(("#", "track"))]
    order = {c: i for i, c in enumerate(CANONICAL)}
    rows.sort(key=lambda r: (order.get(r[0], 99), r[0], int(r[1]), int(r[2])))
    prefix = out[:-4] if out.endswith(".bed") else out
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    paths = {"train": f"{prefix}_train.bed", "test": f"{prefix}_test.bed"}
    counts = {}
    for part, path in paths.items():
        sel = [r for r in rows if (r[3].strip() == BORZOI_TEST_FOLD) == (part == "test")]
        Path(path).write_text("".join("\t".join(r).strip() + "\n" for r in sel), encoding="utf-8")
        counts[part] = len(sel)
    rec = dict(source=src, md5=md5, md5_matches_max_copy=md5 == BORZOI_BED_MD5_REF, rows=len(rows), counts=counts,
               rule=f"test = {BORZOI_TEST_FOLD}; train = all other folds (validation fold4 included)",
               files=paths, date=_dt.datetime.now().isoformat(timespec="seconds"))
    write_json_atomic(Path(f"{prefix}_source.json"), rec)
    print(f"train: {counts['train']}, test: {counts['test']}, md5: {md5}")
    return rec


# ----------------------------------------------------------------------------------------------- CLI
SOURCES = ("flashzoi", "ntv2")


def _chroms(s: Optional[str]) -> Optional[list[str]]:
    return [c for c in s.replace(",", " ").split() if c] if s else None


def main(argv: Optional[Sequence[str]] = None, source_factory: Callable[..., Source] = build_source) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("precompute")
    p.add_argument("--source", required=True, choices=SOURCES)
    p.add_argument("--genome", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chroms", help="comma or space separated; default: the 23 of chr1-22, chrX in the genome")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--weights", help="a local model directory (skips the download; sha256 still checked)")
    p.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
    q = sub.add_parser("project")
    q.add_argument("--dir", required=True)
    q.add_argument("--genome")
    q.add_argument("--fit-chroms", required=True)
    q.add_argument("--dim", type=int, required=True)
    q.add_argument("--chroms")
    q.add_argument("--n-fit", type=int, default=1_000_000)
    q.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("verify")
    v.add_argument("--dir", required=True)
    v.add_argument("--chroms")
    v.add_argument("--genome")
    f = sub.add_parser("folds")
    f.add_argument("--out", required=True, help="prefix: writes <prefix>_train.bed, _test.bed, _source.json")
    f.add_argument("--src", default=BORZOI_BED_URL)
    a = ap.parse_args(argv)
    if a.cmd == "precompute":
        chroms = _chroms(a.chroms) or genome_chroms(Path(a.genome))
        src = source_factory(a.source, a.device, a.weights, a.weights_dir)
        precompute(src, Path(a.genome), Path(a.out), chroms, batch=a.batch)
        return 0
    if a.cmd == "project":
        project(Path(a.dir), Path(a.genome) if a.genome else None, _chroms(a.fit_chroms), a.dim,
                _chroms(a.chroms), n_fit=a.n_fit, seed=a.seed)
        return 0
    if a.cmd == "verify":
        n, ok, bad = verify(Path(a.dir), _chroms(a.chroms), Path(a.genome) if a.genome else None)
        return 0 if ok and bad == 0 else 1
    folds(a.out, a.src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
