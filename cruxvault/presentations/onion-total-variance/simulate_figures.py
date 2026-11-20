"""Simulated illustrations for the total-variance deck. Every number it writes is SIMULATED.

It writes sim_figures.json beside this file. build_deck.py reads that file (and records its sha256).

  (1) two_level   a two-level example: 3 cell types x 40 tracks, Poisson counts of one bin.
                  The law of total variance splits the variance into the variance of the cell-type
                  means and the mean variance inside a cell type.
  (2) chain       the seven-level chain of tools/onion_eval/total_variance.py (its own simulator,
                  unchanged): closed-form terms T_0, T_1..T_6, T_s on counts, the closed-form total, the
                  variance of the simulated counts, the closed-form pair value of every rung and the
                  direct pair value 0.5 mean((Y - Y')^2) of the simulated pairs.
  (3) depth       the closed-form terms at depth factors 0.25 .. 4 (T_s grows as D, T_k as D^2).
  (4) telescope   layers.py run on the simulated tables (halves as measurements), and
                  total_variance.telescoping_check on its rows: the increments add up to the
                  outermost pair value.

Run from the repository root or anywhere:
  /Users/mforooz/miniforge3/envs/candii/bin/python cruxvault/presentations/onion-total-variance/simulate_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tools"))

from onion_eval import total_variance as TV  # noqa: E402

SEED = 0


def two_level(seed=SEED):
    rng = np.random.default_rng(seed)
    means = {"cell type A": 4.0, "cell type B": 10.0, "cell type C": 20.0}   # planted Poisson rates
    groups = {k: rng.poisson(m, 40).astype(float) for k, m in means.items()}
    y = np.concatenate(list(groups.values()))
    gm = {k: float(v.mean()) for k, v in groups.items()}
    between = float(np.mean([(gm[k] - y.mean()) ** 2 for k, v in groups.items() for _ in v]))
    within = float(np.mean([v.var() for v in groups.values()]))
    return dict(planted_rate=means, n_per_group=40,
                values={k: v.astype(int).tolist() for k, v in groups.items()},
                group_mean=gm, group_var={k: float(v.var()) for k, v in groups.items()},
                total=float(y.var()), between=between, within=within,
                identity_abs_diff=abs(float(y.var()) - between - within))


def chain(seed=SEED):
    sim = TV.simulate(TV.BASE_LAYERS, n_cells=6, seed=seed)
    pl = sim["planted"]
    direct = TV.pair_values(sim, "counts")
    return sim, dict(cv2=dict(TV.BASE_LAYERS), n_cells=sim["n_cells"], n_bins=len(sim["base"]),
                     base_mean_reads=3.0, terms=pl["terms"], total=pl["total"],
                     simulated_total=TV.empirical_total(sim), realized_sum=TV.realized_terms(sim)["sum"],
                     pair_value=pl["pair_value"], pair_value_simulated=direct)


def depth(sim):
    out = {}
    for d in (0.25, 0.5, 1.0, 2.0, 4.0):
        p = TV.planted(TV.BASE_LAYERS, sim["base"], sim["n_cells"], depth=d)
        out[f"{d:g}"] = dict(sampling=p["terms"]["sampling"],
                             layers=sum(p["terms"][lv] for lv in TV.LEVELS))
    return out


def telescope(sim):
    got = TV._layers_run(sim, sim["pairs"], "halves", n_boot=100)
    rows = list(got.values())
    chk = TV.telescoping_check(rows)
    lay = {r["rung"]: dict(layer=r["layer"], increment_over=r["increment_over"],
                           variance=float(r["variance"]), increment=float(r["layer_increment"]),
                           depth=r["depth"]) for r in rows}
    return dict(rows=lay, check=[{k: (v if not isinstance(v, (np.floating,)) else float(v))
                                  for k, v in c.items()} for c in chk])


def main():
    sim, ch = chain()
    out = dict(note="SIMULATED: every value in this file comes from a simulation (seed 0); none is real data.",
               seed=SEED, two_level=two_level(), chain=ch, depth=depth(sim), telescope=telescope(sim))
    (HERE / "sim_figures.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    print("wrote", HERE / "sim_figures.json")


if __name__ == "__main__":
    main()
