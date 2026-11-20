import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
R = list(csv.DictReader(open('results/closeout_E4/ladder_two_sided_2026-11-18/two_sided/ladder.tsv'), delimiter='\t'))
def get(cls, rung, stratum='all', band='all'):
    for r in R:
        if r['assay_class'] == cls and r['rung'] == rung and r['stratum'] == stratum and r['band'] == band and r['share'] not in ('', None) and r['flag'].startswith('ok'):
            return float(r['share']), int(r['n_pairs'])
    return None
COL = {'sampling': '#c9ced6', 'replicate': '#e9a13b', 'cell': '#1f6fb5'}
NAMES = {'pseudoreplicate': ('sampling', 'read sampling\n(two halves of one track)'), 'isogenic': ('replicate', 'biological replicate'), 'other_cell_type': ('cell', 'cell type')}
fig = plt.figure(figsize=(13.5, 6.4), dpi=110)
out = {}
for j, (cls, label) in enumerate((('broad_histone', 'Broad histone marks'), ('narrow_histone', 'Narrow histone marks'))):
    ax = fig.add_axes([0.02 + 0.31 * j, 0.10, 0.29, 0.78]); ax.set_aspect('equal'); ax.axis('off')
    v = {k: get(cls, k) for k in NAMES}
    tot = sum(x[0] for x in v.values())
    s = {k: v[k][0] / tot for k in v}
    out[cls] = {k: (round(100 * s[k], 1), v[k][1]) for k in v}
    r_cell = (s['other_cell_type']) ** 0.5; r_rep = (s['other_cell_type'] + s['isogenic']) ** 0.5
    ax.add_patch(Circle((0, 0), 1.0, color=COL['sampling'], ec='white', lw=2))
    ax.add_patch(Circle((0, 0), r_rep, color=COL['replicate'], ec='white', lw=1.5))
    ax.add_patch(Circle((0, 0), r_cell, color=COL['cell'], ec='white', lw=1.5))
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.3, 1.25)
    ax.text(0, 1.12, label, ha='center', va='bottom', fontsize=15, weight='bold')
    ax.text(0, 0.68, f"read sampling\n{100*s['pseudoreplicate']:.0f}%", ha='center', va='center', fontsize=13, color='#222')
    ax.text(0, 0.0, f"cell type\n{100*s['other_cell_type']:.0f}%", ha='center', va='center', fontsize=12, color='white', weight='bold')
    ax.annotate(f"biological replicate {100*s['isogenic']:.1f}%", xy=(r_rep * 0.72, -r_rep * 0.69), xytext=(0.2, -1.18), ha='center', fontsize=12, color='#8a5a0a', arrowprops=dict(arrowstyle='-', color='#8a5a0a', lw=1))
    ax.text(0, -1.3, f"pairs: {v['pseudoreplicate'][1]} read-half, {v['isogenic'][1]} replicate, {v['other_cell_type'][1]} cell-type", ha='center', fontsize=9.5, color='#555')
# right panel: the cell-type layer by genome scale
ax = fig.add_axes([0.70, 0.17, 0.28, 0.62])
bands = [('fine', 'fine\n(25 bp bins)'), ('peak', 'peak\nscale'), ('domain', 'domain\nscale'), ('broad', 'broad\nscale')]
xs, lab = [], []
w = 0.38
for j, (cls, c, name) in enumerate((('broad_histone', '#1f6fb5', 'broad marks'), ('narrow_histone', '#6fb0e6', 'narrow marks'))):
    ys = []
    for b, _ in bands:
        g = get(cls, 'other_cell_type', band=b); ys.append(100 * g[0] if g else float('nan'))
    ax.bar([i + (j - 0.5) * w for i in range(len(bands))], ys, w, color=c, label=name)
    out[cls + '_bands'] = [None if y != y else round(y, 1) for y in ys]
ax.set_xticks(range(len(bands))); ax.set_xticklabels([l for _, l in bands], fontsize=10)
ax.set_ylabel('cell-type layer, percent of the\ncell-type pair value', fontsize=10.5)
ax.set_title('The cell-type layer grows with the scale', fontsize=12.5, weight='bold')
ax.legend(frameon=False, fontsize=10); ax.spines[['top', 'right']].set_visible(False)
fig.text(0.33, 0.955, 'Area of each ring = share of the variance between two cell types (25 bp bins, all regions)', ha='center', fontsize=11.5, color='#333')
fig.savefig('presentations/onion-supervisor-2026-11-18/onion_layers.png', facecolor='white')
import json; print(json.dumps(out))
