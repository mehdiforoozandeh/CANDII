"""Rebuild index.html from deck_src.html by embedding the knob-ranking numbers as DATA.

Paper arc. Input (vault-relative): results/t133/pairs.tsv — one row per (track, arm, level,
space), written by tools/t118/knob_effects.py (Nibi job 25063638). Every number on the slides
is reduced here: per (space, mark class, arm, level) the median over tracks and the min / max
across tracks; per (space, mark class, metric) each knob at its strongest level, ranked.

    python cruxvault/presentations/q9-knob-ranking/build_deck.py
"""
import csv, hashlib, json, pathlib, statistics

HERE = pathlib.Path(__file__).resolve().parent
VAULT = HERE.parent.parent
read = []


def load_tsv(rel):
    p = VAULT / rel
    read.append(p)
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


rows = load_tsv("results/t133/pairs.tsv")
METRICS = ("abs_log2fc_top1", "log2fc_top1", "shape_1k", "dfrip", "frip_base", "frip_arm")

cells = {}
for r in rows:
    k = (r["space"], r["mark_class"], r["arm"], r["level"])
    c = cells.setdefault(k, {m: [] for m in METRICS} | {"tracks": []})
    c["tracks"].append(r["track"])
    for m in METRICS:
        if r[m] not in ("", "None"):
            c[m].append(float(r[m]))


def summ(v):
    return {"v": statistics.median(v), "lo": min(v), "hi": max(v), "n": len(v)} if v else None


levels = []
for (space, mc, arm, level), c in sorted(cells.items()):
    levels.append({"space": space, "mc": mc, "arm": arm, "level": level,
                   **{m: summ(c[m]) for m in METRICS}})

ranked = {}
for space in ("counts", "pval"):
    for mc in ("DNase", "narrow", "broad"):
        for m in ("abs_log2fc_top1", "shape_1k", "dfrip"):
            best = {}
            for L in levels:
                if L["space"] != space or L["mc"] != mc or L[m] is None:
                    continue
                if L["arm"] not in best or abs(L[m]["v"]) > abs(best[L["arm"]][m]["v"]):
                    best[L["arm"]] = L
            order = sorted(best.values(), key=lambda L: -abs(L[m]["v"]))
            ranked[f"{space}|{mc}|{m}"] = [
                {"arm": L["arm"], "level": L["level"], **L[m],
                 "signed": L["log2fc_top1"]["v"] if L["log2fc_top1"] else None} for L in order]

# the base's own setting per mark class, for the "base → arm" label on every bar
MARK_CLASS = {"DNase-seq": "DNase", "H3K27ac": "narrow", "H3K4me3": "narrow", "H3K4me1": "narrow",
              "H3K27me3": "broad", "H3K36me3": "broad", "H3K9me3": "broad"}
base = {}
for r in load_tsv("results/t133/MANIFEST.tsv"):
    if r["arm"] != "base":
        continue
    b = base.setdefault(MARK_CLASS[r["assay"]], {"depth": [], "read_length": [], "fraglen": [], "run_type": []})
    for k in ("depth", "read_length", "fraglen"):
        b[k].append(int(r[k]))
    b["run_type"].append(r["run_type"])
base = {mc: {k: sorted(set(v)) for k, v in b.items()} for mc, b in base.items()}

data = {
    "base": base,
    "n_pairs": len(rows),
    "n_tracks": len({r["track"] for r in rows}),
    "n_arm_products": len({(r["track"], r["arm"], r["level"]) for r in rows}),
    "levels": levels,
    "ranked": ranked,
}

read += [pathlib.Path(__file__).resolve(), HERE / "deck_src.html"]
sources = {p.relative_to(VAULT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in read}
src = (HERE / "deck_src.html").read_bytes().decode("utf-8")
assert src.count("__DATA__") == 1 and src.count("__SOURCES__") == 1
out = src.replace("__SOURCES__", json.dumps(sources, sort_keys=True))
out = out.replace("__DATA__", json.dumps(data, separators=(",", ":")))
(HERE / "index.html").write_bytes(out.encode("utf-8"))
print(f"wrote {HERE / 'index.html'}: {len(sources)} source file(s), {len(rows)} pair rows")
