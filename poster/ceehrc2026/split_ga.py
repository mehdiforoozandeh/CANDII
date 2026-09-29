"""Cut the CANDI graphical abstract into the pieces the poster lays out.

landing/build_ga.py is a verbatim copy of CANDI's docs/build_ga.py (CANDI repo,
commit 4c81728a), the source of https://mehdiforoozandeh.github.io/CANDI/. It is
run unchanged, and its finished figure is then saved again, cropped, once per
piece. Cropping a vector savefig keeps every piece vector and pixel-identical to
the landing page.

Two changes:
- The landing page's shared legend is drawn from the SAGA panel's bars (C), and C
  is not on the poster. That legend is removed, and panel B gets a line legend of
  its own, in panel A's legend style.
- The schematic's tensors and signal tracks are recoloured from teal to the SFU
  red palette (RECOLOUR below). The source is patched in memory before it runs, so
  landing/build_ga.py stays verbatim. The model diagram and panel A keep teal.

Writes panels/ga_{strip,A,B}.{pdf,svg}.
"""
from pathlib import Path

from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
OUT = HERE / "panels"
OUT.mkdir(exist_ok=True)

SFU_RED = "#A6192E"
# (exact text in build_ga.py, replacement); each must match exactly once.
RECOLOUR = [
    # the cubes: measured threads, and imputed ones in the filled cube
    ('            return "#12868C", SHADE[face]', f'            return "{SFU_RED}", SHADE[face]'),
    ('TEAL, TEAL_IMP = "#12868C", "#7FC7C9"', 'TEAL, TEAL_IMP = "#12868C", "#E29AA5"'),
    # the sliced matrices: measured (dark) and imputed (light) signal
    ('["#FFFFFF", "#0E7276"]', '["#FFFFFF", "#8A1426"]'),
    ('["#F4FBFB", "#5FBABD"]', '["#FDF4F5", "#D97A89"]'),
    ('facecolor="#EDF8F8"', 'facecolor="#FBEDEF"'),
    # the predicted tracks: imputed assay names, the 95% band and the mean
    ('color=TEAL if tag else INK', f'color="{SFU_RED}" if tag else INK'),
    ('color=TEAL, alpha=.26', f'color="{SFU_RED}", alpha=.26'),
    ('a.plot(gx, y + mu * BH, color=TEAL', f'a.plot(gx, y + mu * BH, color="{SFU_RED}"'),
    ('Teal names = imputed.', 'Red names = imputed.'),
]
src_path = HERE / "landing" / "build_ga.py"
src = src_path.read_text()
for a, b in RECOLOUR:
    assert src.count(a) == 1, f"not exactly one match: {a}"
    src = src.replace(a, b)
ns = {"__file__": str(src_path), "__name__": "__main__"}
exec(compile(src, str(src_path), "exec"), ns)
fig = ns["fig"]
fig.legends[0].remove()
ns["axB"].legend(fontsize=7.6, frameon=False, loc="lower right", handlelength=1.6,
                 borderpad=0.1, labelspacing=.35)
fig.canvas.draw()
r = fig.canvas.get_renderer()
inch = fig.dpi_scale_trans.inverted()


def box(artists, pad=0.06):
    b = Bbox.union([a.get_tightbbox(r) for a in artists]).transformed(inch)
    return Bbox.from_extents(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad)


strip = [ns["AX"], *ns["AX"].child_axes]
# The strip's own bbox reaches down into the result row, so its bottom edge is cut
# just above the tallest result-panel title.
sb = box(strip)
res_top = max(a.get_tightbbox(r).transformed(inch).y1 for a in
              (ns["axA"], ns["axB"], ns["axC"]))
sb = Bbox.from_extents(sb.x0, res_top + 0.04, sb.x1, sb.y1)
pieces = {
    "ga_strip": sb,
    "ga_A": box([ns["axA"]]),
    "ga_B": box([ns["axB"]]),
}
for name, bb in pieces.items():
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches=bb, facecolor="white")
    print(f"wrote panels/{name}.{{pdf,svg}}  ({bb.width:.2f} x {bb.height:.2f} in)")
