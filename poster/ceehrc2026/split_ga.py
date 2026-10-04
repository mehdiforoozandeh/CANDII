"""Cut the poster's CANDI schematic into the pieces the poster lays out.

schematic.py draws the figure; it began as landing/build_ga.py, a verbatim copy
of CANDI's docs/build_ga.py (CANDI repo, commit 4c81728a), the source of
https://mehdiforoozandeh.github.io/CANDI/. Its finished figure is saved again,
cropped, once per piece, so every piece stays vector.

The landing page's shared legend is drawn from the SAGA panel's bars (C), and C
is not on the poster. That legend is removed, and panel B gets a line legend of
its own, in panel A's legend style.

Writes panels/ga_{strip,A,B}.{pdf,svg}.
"""
from pathlib import Path

from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
OUT = HERE / "panels"
OUT.mkdir(exist_ok=True)

src_path = HERE / "schematic.py"
ns = {"__file__": str(src_path), "__name__": "__main__"}
exec(compile(src_path.read_text(), str(src_path), "exec"), ns)
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
