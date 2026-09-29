"""Cut the CANDI graphical abstract into the pieces the poster lays out.

landing/build_ga.py is a verbatim copy of CANDI's docs/build_ga.py (CANDI repo,
commit 4c81728a), the source of https://mehdiforoozandeh.github.io/CANDI/. It is
run unchanged, and its finished figure is then saved again, cropped, once per
piece. Cropping a vector savefig keeps every piece vector and pixel-identical to
the landing page.

Writes panels/ga_{strip,A,B,legend}.pdf. The SAGA panel (C) is not used.
"""
from pathlib import Path
import runpy

from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
OUT = HERE / "panels"
OUT.mkdir(exist_ok=True)

ns = runpy.run_path(str(HERE / "landing" / "build_ga.py"))
fig = ns["fig"]
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
    "ga_legend": box([fig.legends[0]]),
}
for name, bb in pieces.items():
    fig.savefig(OUT / f"{name}.pdf", bbox_inches=bb, facecolor="white")
    print(f"wrote panels/{name}.pdf  ({bb.width:.2f} x {bb.height:.2f} in)")
