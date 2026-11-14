#!/usr/bin/env python3
"""Frozen sequence-model embeddings on the 25 bp grid, for the DNA source arms (`dna_flashzoi`, `dna_gnlm`).

Runs in the separate DNA venv (`requirements-dnaemb.txt`, built by `slurm/minicandi/dna_embed.sh`), never in the main
kit venv. It imports only numpy and h5py at module level; torch, borzoi_pytorch and transformers load only when a real
model is built. It imports nothing from `candi` or `minicandi`, so it runs in that venv as is.

    dna_embed.py precompute --source flashzoi|ntv2 --genome $O/genome --out <dir> --chroms chr21 [--device cuda]
    dna_embed.py fit        --source flashzoi|ntv2 --genome $O/genome --out <projection.npz> --fit-chroms chr1 --dim 128
                            [--fit-bed <bed>] [--exclude-bed <bed>] [--n-fit 1000000] [--per-window 2000]
    dna_embed.py precompute ... --projection <projection.npz>     (streamed: only projected files are written)
    dna_embed.py project    --dir <dir> --fit-chroms chr1 --dim 128 [--chroms ...]
    dna_embed.py verify     --dir <dir> [--chroms ...] [--genome <genome dir>]
    dna_embed.py folds      --out <prefix> [--src <url or path of sequences_human.bed.gz>]

Output of `precompute`: `<dir>/<chrom>.h5` (dataset `emb`, float16 `[n_bins, d]`, chunks of 4096 bins; n_bins is the
length of the genome layer's `mask.h5[chrom]`, i.e. floor(chrom length / 25)) and `<dir>/meta.json`. `project` fits a
PCA on random valid bins (mask.h5 = 1) of the fit chromosomes (no label is read), writes `<dir>/projection.npz` and
replaces each `<chrom>.h5` by its projection in place (the unprojected file is the deleted intermediate).
The genome-wide route avoids that intermediate: `fit` runs the model on random windows of the fit chromosomes and fits
the PCA from the streamed embeddings (no file but `projection.npz`), and `precompute --projection` projects each batch
before it is written, so only the `[n_bins, dim]` files persist. Both routes give the same values (test).

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
def window_bins(src: Source, codes: np.ndarray, idx: np.ndarray, wins: Sequence[int], batch: int):
    """Run the model on windows `wins` (in batches) and yield (w, b0, b1, z) per window: z float32 [b1 - b0, d] holds
    the embeddings of the 25 bp bins b0..b1 - 1 whose centre lies in window w's kept span."""
    per_win = src.out_bp // src.native_bp
    wins = list(wins)
    for i in range(0, len(wins), batch):
        ws = wins[i:i + batch]
        x = np.stack([padded(codes, w * src.out_bp - src.crop_bp, src.input_bp) for w in ws])
        z = np.asarray(src.embed(x), np.float32)
        if z.shape != (len(ws), per_win, src.d):
            raise ValueError(f"{src.name}: embed gave {z.shape}, expected {(len(ws), per_win, src.d)}")
        for j, w in enumerate(ws):
            n0 = w * per_win
            b0, b1 = int(np.searchsorted(idx, n0, "left")), int(np.searchsorted(idx, n0 + per_win, "left"))
            yield w, b0, b1, z[j][idx[b0:b1] - n0]


def n_windows(src: Source, n_bins: int) -> tuple[np.ndarray, int]:
    """Native index of each 25 bp bin, and the number of model windows that cover the chromosome."""
    idx = native_index(n_bins, src.native_bp)
    n_native = int(idx[-1]) + 1 if n_bins else 0
    return idx, -(-n_native // (src.out_bp // src.native_bp))


def load_projection(path: Path) -> dict:
    """projection.npz -> its arrays, with mean and components as float32 and the file's sha256."""
    p = dict(np.load(path, allow_pickle=False))
    p["mean"] = np.asarray(p["mean"], np.float32)
    p["components"] = np.asarray(p["components"], np.float32)
    p["sha256"] = sha256_file(Path(path))
    return p


def apply_projection(z: np.ndarray, proj: dict) -> np.ndarray:
    """The projection `project` applies: the float16-rounded embedding, minus the mean, times the components; float16
    out. Rounding z to float16 first makes a streamed projection equal precompute-then-project."""
    z = np.asarray(z, np.float32).astype(np.float16).astype(np.float32)
    return ((z - proj["mean"]) @ proj["components"].T).astype(np.float16)


def precompute_chrom(src: Source, genome: Path, chrom: str, out: Path, batch: int = 4,
                     log: Callable[[str], None] = print, proj: Optional[dict] = None) -> Path:
    """Write `<out>/<chrom>.h5`; skips a chromosome whose file exists (a crash leaves only `<chrom>.h5.tmp`).

    With `proj` (from `load_projection`), each batch is projected as it is computed and only the projected file is
    written: no unprojected intermediate exists on disk."""
    src.check()
    final = Path(out) / f"{chrom}.h5"
    dim = int(proj["components"].shape[0]) if proj is not None else 0
    if final.is_file():
        with h5py.File(final, "r") as h:
            pdim, psha = int(h.attrs.get("projected_dim", 0)), str(h.attrs.get("projection_sha256", ""))
        if proj is not None and (pdim != dim or psha != proj["sha256"]):
            raise SystemExit(f"{final} exists with projected_dim {pdim} or another projection; move it away first")
        log(f"[dna_embed] {chrom}: {final} exists, skipped")
        return final
    if proj is not None and proj["components"].shape[1] != src.d:
        raise SystemExit(f"projection d_in {proj['components'].shape[1]} is not the model's d {src.d}")
    n_bins = genome_bins(genome, chrom)
    with h5py.File(Path(genome) / "dna.h5", "r") as f:
        codes = np.asarray(f[chrom][:], np.uint8)
    idx, n_win = n_windows(src, n_bins)
    d_out = dim or src.d
    tmp = final.with_name(final.name + ".tmp")
    with h5py.File(tmp, "w") as h:
        ds = h.create_dataset("emb", shape=(n_bins, d_out), dtype="float16",
                              chunks=(max(1, min(CHUNK_BINS, n_bins)), d_out))
        h.attrs.update(source=src.name, chrom=chrom, native_bp=src.native_bp, res=RES, projected_dim=dim)
        if proj is not None:
            h.attrs.update(projection_sha256=proj["sha256"], d_in=int(src.d))
        for w, b0, b1, zb in window_bins(src, codes, idx, range(n_win), batch):
            if b1 > b0:
                ds[b0:b1] = apply_projection(zb, proj) if proj is not None else zb.astype(np.float16)
            if w % (batch * 500) == 0:
                log(f"[dna_embed] {chrom}: window {w}/{n_win} {_dt.datetime.now().isoformat(timespec='seconds')}")
    os.replace(tmp, final)
    log(f"[dna_embed] {chrom}: {n_bins} bins x {d_out} -> {final}")
    return final


def projection_meta(proj: dict) -> dict:
    rec = dict(method="pca", dim=int(proj["components"].shape[0]), d_in=int(proj["components"].shape[1]),
               file="projection.npz", sha256=proj["sha256"], applied="streamed inside precompute (no intermediate)")
    for k in ("n_fit", "seed", "explained", "sample", "fit_bed", "exclude_bed", "fit_bed_md5", "exclude_bed_md5",
              "n_windows", "per_window"):
        if k in proj:
            rec[k] = np.asarray(proj[k]).item()
    if "fit_chroms" in proj:
        rec["fit_chroms"] = [str(c) for c in np.atleast_1d(proj["fit_chroms"])]
    return rec


def precompute(src: Source, genome: Path, out: Path, chroms: Sequence[str], batch: int = 4,
               log: Callable[[str], None] = print, projection: Optional[Path] = None) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    proj = load_projection(Path(projection)) if projection else None
    meta = read_meta(out)
    if meta and meta.get("source") != src.name:
        raise SystemExit(f"{out}/meta.json holds source {meta.get('source')!r}, not {src.name!r}")
    meta.update(source=src.name, native_bin_bp=src.native_bp, receptive_field_bp=src.input_bp,
                kept_span_bp=src.out_bp, d=src.d, res=RES, genome=str(Path(genome).resolve()),
                grid_rule="25 bp bin b takes native bin floor((25 b + 12.5) / native_bp)",
                dtype="float16", chunk_bins=CHUNK_BINS, date=_dt.date.today().isoformat(), **src.meta)
    meta.setdefault("projection", None)
    if proj is not None:
        meta.update(projection=projection_meta(proj), d=int(proj["components"].shape[0]), d_model=int(src.d),
                    projection_path=str(Path(projection).resolve()))
    write_json_atomic(out / "meta.json", meta)
    for c in chroms:
        precompute_chrom(src, genome, c, out, batch=batch, log=log, proj=proj)


# ----------------------------------------------------------------------------------------------- fit (streamed PCA)
def read_bed(path: Optional[str]) -> dict:
    """chrom -> (sorted starts, sorted ends) int64 arrays; overlapping intervals are allowed."""
    if not path:
        return {}
    acc: dict = {}
    for ln in Path(path).read_text().splitlines():
        if not ln.strip() or ln.startswith(("#", "track")):
            continue
        c, s, e = ln.split("\t")[:3]
        acc.setdefault(c, ([], []))
        acc[c][0].append(int(s))
        acc[c][1].append(int(e))
    return {c: (np.sort(np.array(s, np.int64)), np.sort(np.array(e, np.int64))) for c, (s, e) in acc.items()}


def bins_in(bed: dict, chrom: str, n_bins: int) -> np.ndarray:
    """Bool [n_bins]: the 25 bp bin's centre lies in at least one interval [start, end) of `bed` on `chrom`.
    (#intervals with start <= centre) - (#intervals with end <= centre) = #intervals holding the centre."""
    if chrom not in bed:
        return np.zeros(n_bins, bool)
    s, e = bed[chrom]
    c2 = 2 * RES * np.arange(n_bins, dtype=np.int64) + RES                   # 2 x centre, exact in integers
    return (np.searchsorted(2 * s, c2, "right") - np.searchsorted(2 * e, c2, "right")) > 0


def fit_streamed(src: Source, genome: Path, fit_chroms: Sequence[str], dim: int, n_fit: int = 1_000_000,
                 seed: int = 0, fit_bed: Optional[str] = None, exclude_bed: Optional[str] = None,
                 per_window: int = 2000, batch: int = 4, log: Callable[[str], None] = print) -> dict:
    """PCA of up to n_fit bins, drawn by running the model on random windows of the fit chromosomes; no embedding
    file is written. Eligible bins: mask.h5 = 1, centre inside `fit_bed` (if given) and outside `exclude_bed` (if
    given). Windows holding an eligible bin are visited in a seeded random order; each gives at most `per_window`
    random eligible bins. Mean and covariance accumulate in float64 from the float16-rounded embeddings."""
    src.check()
    rng = np.random.default_rng(seed)
    inc, exc = read_bed(fit_bed), read_bed(exclude_bed)
    per = src.out_bp // src.native_bp
    plan, elig, idxs = [], {}, {}
    for c in fit_chroms:
        nb = genome_bins(genome, c)
        with h5py.File(Path(genome) / "mask.h5", "r") as f:
            ok = np.asarray(f[c][:]) > 0
        if fit_bed:
            ok &= bins_in(inc, c, nb)
        if exclude_bed:
            ok &= ~bins_in(exc, c, nb)
        idx, n_win = n_windows(src, nb)
        elig[c], idxs[c] = ok, idx
        cs = np.concatenate([[0], np.cumsum(ok, dtype=np.int64)])
        wb = np.searchsorted(idx, np.arange(n_win + 1) * per, "left")
        plan += [(c, int(w)) for w in np.flatnonzero(cs[wb[1:]] - cs[wb[:-1]] > 0)]
    order = [plan[i] for i in rng.permutation(len(plan))]
    s1 = np.zeros(src.d, np.float64)
    s2 = np.zeros((src.d, src.d), np.float64)
    n = used = pos = 0
    codes_c, codes = None, None
    while n < n_fit and pos < len(order):
        c = order[pos][0]
        ws = []
        while pos < len(order) and order[pos][0] == c and len(ws) < batch:
            ws.append(order[pos][1])
            pos += 1
        if codes_c != c:
            with h5py.File(Path(genome) / "dna.h5", "r") as f:
                codes = np.asarray(f[c][:], np.uint8)
            codes_c = c
        for w, b0, b1, zb in window_bins(src, codes, idxs[c], ws, batch):
            rows = np.flatnonzero(elig[c][b0:b1])
            take = min(per_window, rows.size, n_fit - n)
            if take <= 0:
                continue
            rows = np.sort(rng.choice(rows, size=take, replace=False))
            x = zb[rows].astype(np.float16).astype(np.float64)
            s1 += x.sum(0)
            s2 += x.T @ x
            n += take
            used += 1
            if used % 100 == 0:
                log(f"[dna_embed] fit: {n}/{n_fit} bins from {used} windows "
                    f"{_dt.datetime.now().isoformat(timespec='seconds')}")
    if n <= dim:
        raise SystemExit(f"fit: only {n} eligible bins for dim {dim}")
    mean = s1 / n
    cov = (s2 - n * np.outer(mean, mean)) / (n - 1)
    ev, v = np.linalg.eigh(cov)
    top = np.argsort(ev)[::-1][:dim]
    comp = v[:, top].T
    sign = np.sign(comp[np.arange(dim), np.abs(comp).argmax(1)])
    comp = comp * np.where(sign == 0, 1, sign)[:, None]

    def md5(p):
        return hashlib.md5(Path(p).read_bytes()).hexdigest() if p else ""
    return dict(mean=mean.astype(np.float32), components=comp.astype(np.float32),
                eigenvalues=ev[top].astype(np.float64), explained=float(ev[top].sum() / max(ev.sum(), 1e-300)),
                n_fit=int(n), seed=int(seed), fit_chroms=list(fit_chroms), n_windows=int(used),
                per_window=int(per_window), fit_bed=str(fit_bed or ""), exclude_bed=str(exclude_bed or ""),
                fit_bed_md5=md5(fit_bed), exclude_bed_md5=md5(exclude_bed),
                sample=("mask.h5 = 1 bins" + (", centre in fit_bed" if fit_bed else "")
                        + (", centre not in exclude_bed" if exclude_bed else "")
                        + "; random windows, at most per_window random bins each; no label read"))


def fit(src: Source, genome: Path, out_npz: Path, fit_chroms: Sequence[str], dim: int, **kw) -> Path:
    """Fit the streamed PCA and write it to `out_npz` (refuses to overwrite an existing projection)."""
    out_npz = Path(out_npz)
    if out_npz.is_file():
        raise SystemExit(f"{out_npz} exists; a projection is never refitted in place")
    p = fit_streamed(src, genome, fit_chroms, dim, **kw)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npz.with_name(f".{out_npz.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in p.items()})
    os.replace(tmp, out_npz)
    print(f"[dna_embed] fit: PCA {dim} of {src.d} on {p['n_fit']} bins from {p['n_windows']} windows: "
          f"explained {p['explained']:.4f} -> {out_npz}")
    return out_npz


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
    p.add_argument("--projection", help="a projection.npz from `fit`: project each batch as it is computed and "
                                        "write only the projected file")
    t = sub.add_parser("fit")
    for k in ("--source", "--genome", "--out"):
        t.add_argument(k, required=True, **({"choices": SOURCES} if k == "--source" else {}))
    t.add_argument("--fit-chroms", required=True)
    t.add_argument("--dim", type=int, required=True)
    t.add_argument("--n-fit", type=int, default=1_000_000)
    t.add_argument("--per-window", type=int, default=2000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--fit-bed", help="only bins whose centre is in these intervals")
    t.add_argument("--exclude-bed", help="never bins whose centre is in these intervals")
    t.add_argument("--device", default="cuda")
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--weights")
    t.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
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
        precompute(src, Path(a.genome), Path(a.out), chroms, batch=a.batch, projection=a.projection)
        return 0
    if a.cmd == "fit":
        src = source_factory(a.source, a.device, a.weights, a.weights_dir)
        fit(src, Path(a.genome), Path(a.out), _chroms(a.fit_chroms), a.dim, n_fit=a.n_fit, seed=a.seed,
            fit_bed=a.fit_bed, exclude_bed=a.exclude_bed, per_window=a.per_window, batch=a.batch)
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
