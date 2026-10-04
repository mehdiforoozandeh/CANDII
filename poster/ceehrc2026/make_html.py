"""The poster as a self-contained web page.

Writes html/poster_landscape.html (44 x 34 in). The page mirrors
poster_landscape.tex: it is laid out in CSS inches and points at print size,
then scaled to the window's width. Every sentence is read from common.tex, so
the words cannot drift from the PDF; every panel is the SVG twin of the PDF the
LaTeX includes, embedded as a data URI, so the page is one file.

Run after split_ga.py, make_eic_panels.py, make_intro_panels.py and the qr.tex
build (build.sh does).
"""
from base64 import b64encode
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PANELS, OUT = HERE / "panels", HERE / "html"
OUT.mkdir(exist_ok=True)


# ------------------------------------------------------------ the words ------
def macros(tex):
    """Every \\newcommand{\\Name}{body} in tex, with nested braces."""
    out = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{", tex):
        i, depth = m.end(), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(tex[i], 0) if tex[i - 1] != "\\" else 0
            i += 1
        out[m.group(1)] = tex[m.end():i - 1]
    return out


# The only inline maths on the poster. A new formula in common.tex fails here
# until it has an HTML form.
MATH = {
    r"$\times$": "×",
    r"$r \approx 0.79$": "<i>r</i> ≈ 0.79",
    r"$r$": "<i>r</i>",
}


def html(tex):
    s = re.sub(r"%\n\s*", "", tex.strip())            # line-end comments
    s = re.sub(r"\s+", " ", s)
    for k, v in MATH.items():
        s = s.replace(k, v)
    assert "$" not in s, f"unmapped maths in: {s}"
    s = s.replace(r"\%", "%").replace("--", "–").replace("~", "&nbsp;")
    s = s.replace(r"\,", "&thinsp;").replace(r"\enspace", "&ensp;").replace(r"\quad", "&emsp;")
    for cmd, tag in (("textbf", "b"), ("textit", "i"), ("emph", "em")):
        s = re.sub(rf"\\{cmd}\{{([^{{}}]*)\}}", rf"<{tag}>\1</{tag}>", s)
    s = re.sub(r"\\par\s*\\vspace\{[^}]*\}\s*", "</p><p>", s)
    s = re.sub(r"\\par\s*", "</p><p>", s)
    s = s.replace(r"\begin{itemize}", "<ul>").replace(r"\end{itemize}", "</ul>")
    s = re.sub(r"\\item\s*", "<li>", s)
    s = s.replace("<li>", "</li><li>").replace("<ul></li>", "<ul>")
    s = s.replace(" </ul>", "</ul>").replace("</ul>", "</li></ul>")
    assert "\\" not in s, f"unconverted LaTeX in: {s}"
    return s


W = {k: html(v) for k, v in macros((HERE / "common.tex").read_text()).items()
     if k[:3] in ("Txt", "Cap", "Hea", "Pos")}


def svg(stem):
    data = b64encode((PANELS / f"{stem}.svg").read_bytes()).decode()
    return f"data:image/svg+xml;base64,{data}"


# ------------------------------------------------------------- the page ------
CSS = """
:root {
  --ground: #E7EBED; --ground-ink: #3C4A52;
  --red: #A6192E; --heading: #2E4A62; --ink: #1B2A32; --muted: #5E6E78; --panel: #F3F6F9;
  --rule: #D9E1E8; --paper: #FFFFFF; --accent: #A9BFD3;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --ground: #161C20; --ground-ink: #AAB6BD; color-scheme: dark; }
}
:root[data-theme="dark"] { --ground: #161C20; --ground-ink: #AAB6BD; color-scheme: dark; }
body { background: var(--ground); color: var(--ground-ink);
  font-family: "Open Sans", "Helvetica Neue", Arial, sans-serif;
  padding-inline: 16px; padding-block: 14px 24px; }
.bar { max-width: 1800px; margin: 0 auto 10px; font-size: 13px; letter-spacing: .02em; }
.wrap { max-width: 1800px; margin: 0 auto; overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,.18), 0 8px 28px rgba(0,0,0,.12); }
/* The poster itself: real inches and points, then scaled by the script. The
   paper stays white in both themes, as it prints. */
.poster { position: relative; background: var(--paper); color: var(--ink);
  transform-origin: 0 0; font-family: "Open Sans", "Helvetica Neue", Arial, sans-serif; }
.poster * { box-sizing: border-box; margin: 0; }
.poster p + p { margin-top: .1in; }
.red { position: absolute; left: .8in; top: var(--red-top); width: var(--red-w); height: var(--red-h);
  background: var(--red); }
.sfu { position: absolute; right: var(--sfu-r); bottom: -.02in; color: #fff; line-height: .8;
  font-size: var(--sfu); font-weight: 400; }
.title { position: absolute; left: var(--title-x); font-weight: 700; white-space: nowrap;
  line-height: 1; }
.authors { position: absolute; left: var(--title-x); top: var(--authors-top);
  font-size: var(--authors); line-height: 1; white-space: nowrap; }
.affil { position: absolute; left: var(--title-x); top: var(--affil-top);
  font-size: 38pt; line-height: 1; white-space: nowrap; color: var(--muted); }
/* The QR code at the right margin, its link under it on the affiliation's line. */
.qr-col { position: absolute; right: .8in; top: .6in; display: flex; flex-direction: column;
  align-items: flex-end; font-size: 20.4pt; line-height: 1; color: var(--muted); white-space: nowrap; }
.qr-col img { display: block; width: 1.74in; height: 1.74in; margin-bottom: var(--url-gap); }
.divider { position: absolute; left: .8in; right: .8in; top: 3.25in; height: 4pt;
  background: var(--rule); }
.body { position: absolute; left: .8in; top: var(--top); display: grid; }
.col { display: flex; flex-direction: column; min-height: 0; }
.row { display: flex; justify-content: space-between; align-items: flex-start; }
.fill { flex: 1; }
.intro { font-size: var(--body); line-height: var(--body-lh); }
.intro ul { list-style: none; padding: 0; }
.intro li { position: relative; padding-left: 1em; }
.intro li + li { margin-top: .15in; }
.intro li::before { content: "•"; color: var(--accent); position: absolute; left: .15em; }
.poster h2 { color: var(--heading); font-size: var(--head); line-height: 1.2; font-weight: 700;
  letter-spacing: -.01em; text-wrap: balance; padding-bottom: .08in; border-bottom: 5pt solid var(--accent);
  margin-bottom: .3in; }
figure img { display: block; width: 100%; }
.poster figcaption, .poster .cap { font-size: var(--cap); line-height: var(--cap-lh); margin-top: .2in; }
.cap.wide { text-align: justify; margin-top: .18in; }
.uses { display: grid; grid-template-columns: 5.7in 5.7in; justify-content: space-between;
  row-gap: .15in; }
.refs { border-top: 4pt solid var(--rule); margin-top: .15in; padding-top: .15in; font-size: var(--small);
  line-height: var(--small-lh); color: var(--muted); }
.poster .refs p + p { margin-top: 0; }
sub, sup { font-size: .7em; line-height: 0; }
@media (prefers-reduced-motion: no-preference) { .wrap { transition: height .15s; } }
"""

SCRIPT = """
<script>
(function () {
  const wrap = document.querySelector('.wrap'), poster = document.querySelector('.poster');
  function fitTitle() {            // the LaTeX \\resizebox: the title fills its width
    document.querySelectorAll('[data-fit]').forEach(el => {
      const want = parseFloat(el.dataset.fit) * 96, f = parseFloat(getComputedStyle(el).fontSize);
      el.style.fontSize = (f * want / el.scrollWidth) + 'px';
    });
  }
  function fit() {
    const s = wrap.clientWidth / poster.offsetWidth;
    poster.style.transform = 'scale(' + s + ')';
    wrap.style.height = (poster.offsetHeight * s) + 'px';
  }
  fitTitle(); fit();
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(() => { fitTitle(); fit(); });
  new ResizeObserver(fit).observe(wrap);
})();
</script>
"""


def fig(src, width, cap):
    return (f'<figure style="width:{width}in"><img src="{svg(src)}" alt="">'
            f'<figcaption>{cap}</figcaption></figure>')


def page(kind, w_in, h_in, sizes, header, body):
    vars_ = "; ".join(f"--{k}: {v}" for k, v in sizes.items())
    return f"""<meta charset="utf-8">
<title>CANDI CEEHRC poster ({kind})</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;700&display=swap">
<style>{CSS}</style>
<div class="bar">CEEHRC 2026 poster · {kind}, {w_in:g} × {h_in:g} in · scaled to the window; zoom in to read at print size</div>
<div class="wrap">
<div class="poster" style="width:{w_in}in; height:{h_in}in; {vars_}">
{header}
{body}
</div>
</div>
{SCRIPT}"""


def qr():
    link = f'<a href="{W['PosterURL']}" style="color:inherit">mehdiforoozandeh.github.io/CANDI</a>'
    img = f'<img src="{svg('qr')}" alt="QR code: {W['PosterURL']}">'
    return f'<div class="qr-col">{img}{link}</div>'


def refs():
    return f'<div class="refs"><p>{W["TxtRefs"]}</p></div>'


# --- poster_landscape.tex, 44 x 36 in (fixed) -------------------------------
L = page("landscape", 44, 36,
         dict(**{"red-w": "5.6in", "red-top": ".6in", "red-h": "2.35in", "sfu-r": ".21in",
                 "sfu": "112pt", "title-x": "7.1in", "authors": "52pt",
                 "authors-top": "1.575in", "affil-top": "2.487in", "url-gap": ".36in",
                 "top": "3.55in", "body": "28pt", "body-lh": "35pt", "cap": "26pt",
                 "cap-lh": "34pt", "small": "20pt", "small-lh": "26pt", "head": "48pt"}),
         f"""<div class="red"><div class="sfu">SFU</div></div>
<div class="title" data-fit="33.7" style="top:.5in; font-size:100pt">{W['PosterTitle']}</div>
<div class="authors">{W['PosterAuthors']}</div>
<div class="affil">{W['PosterAffil']}</div>
{qr()}
<div class="divider"></div>""",
         f"""<div class="body" style="grid-template-rows: 15.96in 13.65in auto; width: 42.4in">
<div style="display:grid; grid-template-columns: 12.2in 29.2in; column-gap: 1in">
<div class="col"><h2>{W['HeadProblem']}</h2>
<div class="intro">{W['TxtProblem']}</div>
<div class="fill"></div>
<h2>{W['HeadUses']}</h2>
<div class="uses">{''.join(fig(f'use_{s}', 5.7, W[t]) for s, t in
     (('states', 'TxtUseStates'), ('gwas', 'TxtUseGwas'), ('expr', 'TxtUseExpr'),
      ('conf', 'TxtUseConf')))}</div>
</div>
<div class="col"><h2>{W['HeadStrip']}</h2>
<img src="{svg('ga_strip')}" alt="CANDI schematic workflow" style="width:100%">
<p class="cap wide">{W['CapStrip']}</p></div>
</div>
<div style="display:grid; grid-template-columns: 28.4in 13in; column-gap: 1in; margin-top: .4in">
<div class="col"><h2>{W['HeadEIC']}</h2>
<div class="row">{fig('eic_pearson_landscape', 7.0, W['CapA'])}
{fig('eic_skill_landscape', 8.8, W['CapB'])}
{fig('eic_measures_landscape', 11.8, W['CapC'])}</div></div>
<div class="col"><h2>{W['HeadUtility']}</h2>
<div class="fill"></div>
<div class="row" style="align-items:center"><img src="{svg('ga_A')}" alt="" style="width:7in">
<p class="cap" style="width:5.6in; margin:0">{W['CapD']}</p></div>
<div class="fill"></div>
<div class="row" style="align-items:center"><img src="{svg('ga_B')}" alt="" style="width:7in">
<p class="cap" style="width:5.6in; margin:0">{W['CapE']}</p></div>
<div class="fill"></div></div>
</div>
{refs()}
</div>""")

for name, text in (("poster_landscape", L),):
    f = OUT / f"{name}.html"
    f.write_text(text)
    print(f"wrote html/{f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
