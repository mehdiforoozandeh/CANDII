"""The poster as two self-contained web pages, one per layout.

Writes html/poster_landscape.html (44 x 34 in) and html/poster_portrait.html
(34 x 44 in). Each page mirrors its LaTeX layout (poster_landscape.tex,
poster_portrait.tex): it is laid out in CSS inches and points at print size,
then scaled to the window's width. Every sentence is read from common.tex, so
the words cannot drift from the PDFs; every panel is the SVG twin of the PDF
the LaTeX includes, embedded as a data URI, so each page is one file.

Run after split_ga.py, make_eic_panels.py and the qr.tex build (build.sh does).
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
    r"$9\times10^{-10}$": "9&thinsp;×&thinsp;10<sup>−10</sup>",
    r"$\max(\beta_0 + \beta_1\cdot\text{CANDI} + \beta_2\cdot\text{baseline},\,0)$":
        "max(<i>β</i><sub>0</sub> + <i>β</i><sub>1</sub>·CANDI + "
        "<i>β</i><sub>2</sub>·baseline, 0)",
    r"$r \approx 0.79$": "<i>r</i> ≈ 0.79",
}


def html(tex):
    s = re.sub(r"%\n\s*", "", tex.strip())            # line-end comments
    s = re.sub(r"\s+", " ", s)
    for k, v in MATH.items():
        s = s.replace(k, v)
    assert "$" not in s, f"unmapped maths in: {s}"
    s = s.replace(r"\%", "%").replace("--", "–").replace("~", "&nbsp;")
    s = s.replace(r"\,", "&thinsp;").replace(r"\enspace", "&ensp;")
    for cmd, tag in (("textbf", "b"), ("textit", "i"), ("emph", "em")):
        s = re.sub(rf"\\{cmd}\{{([^{{}}]*)\}}", rf"<{tag}>\1</{tag}>", s)
    s = re.sub(r"\s*\\centerline\{([^{}]*)\}\s*\\vspace\{[^}]*\}\s*",
               r'</p><p class="eq">\1</p><p>', s)
    s = re.sub(r"\\par\s*\\vspace\{[^}]*\}\s*", "</p><p>", s)
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
  --red: #A6192E; --ink: #1B2A32; --muted: #5E6E78; --panel: #FCF4F1;
  --rule: #EBD9D4; --paper: #FFFFFF; --accent: #E8A598;
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
.poster .text p + p { margin-top: .2in; }
.poster .text p.eq, .poster .text p.eq + p { margin-top: .12in; }
.poster p.eq { text-align: center; white-space: nowrap; }
.red { position: absolute; left: 0; top: .6in; width: var(--red-w); height: 3.35in;
  background: var(--red); }
.sfu { position: absolute; right: .3in; bottom: -.02in; color: #fff; line-height: .8;
  font-size: var(--sfu); font-weight: 400; }
.title { position: absolute; left: var(--title-x); font-weight: 700; white-space: nowrap;
  line-height: 1.12; }
.authors { position: absolute; left: var(--title-x); bottom: calc(100% - 3.8in);
  font-size: var(--authors); line-height: 1; white-space: nowrap; }
.qr { position: absolute; right: .8in; top: .6in; display: flex; flex-direction: column;
  align-items: flex-end; white-space: nowrap;
  font-size: 17pt; color: var(--muted); }
.qr img { display: block; width: 2.9in; height: 2.9in; margin-bottom: .1in; }
/* Landscape: a small QR code at the right end of the author line. */
.qr-line { position: absolute; right: .8in; bottom: calc(100% - 3.8in); display: flex;
  align-items: flex-end; gap: .2in; font-size: 17pt; color: var(--muted); white-space: nowrap; }
.qr-line img { display: block; width: 1.45in; height: 1.45in; }
.body { position: absolute; left: .8in; top: var(--top); display: grid; }
.col { display: flex; flex-direction: column; min-height: 0; }
.row { display: flex; justify-content: space-between; align-items: flex-start; }
.fill { flex: 1; }
.box { background: var(--panel); border: 4pt solid var(--accent); border-radius: .22in;
  padding: .28in .4in .28in;
  font-size: var(--body); line-height: var(--body-lh); text-align: justify; hyphens: auto; }
.box h3 { color: var(--red); font-size: var(--boxtitle); line-height: 1.2;
  font-weight: 700; margin-bottom: .15in; }
.box ul { list-style: none; padding: 0; }
.box li { position: relative; padding-left: 1em; }
.box li + li { margin-top: .15in; }
.box li::before { content: "•"; color: var(--accent); position: absolute; left: .15em; }
.poster h2 { color: var(--red); font-size: var(--head); line-height: 1.2; font-weight: 700;
  text-wrap: balance; padding-bottom: .08in; border-bottom: 5pt solid var(--accent);
  margin-bottom: .3in; }
figure img { display: block; width: 100%; }
.poster figcaption, .poster .cap { font-size: var(--cap); line-height: var(--cap-lh); margin-top: .2in; }
.cap.wide { text-align: justify; margin-top: .18in; }
.text { font-size: var(--body); line-height: var(--body-lh); text-align: justify; hyphens: auto; }
.refs { border-top: 3pt solid var(--rule); padding-top: .25in; font-size: var(--small);
  line-height: var(--small-lh); color: var(--muted); }
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


def box(title, text, height=None):
    h = f' style="height:{height}in"' if height else ""
    return f'<div class="box"{h}><h3>{title}</h3>{text}</div>'


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


def qr(cls):
    link = f'<a href="{W['PosterURL']}" style="color:inherit">mehdiforoozandeh.github.io/CANDI</a>'
    img = f'<img src="{svg('qr')}" alt="QR code: {W['PosterURL']}">'
    return f'<div class="{cls}">{img if cls == "qr" else link}{link if cls == "qr" else img}</div>'


def refs():
    return f'<div class="refs"><p>{W["TxtRefs"]}</p></div>'


# --- landscape: poster_landscape.tex, 44 x 34 in -----------------------------
L = page("landscape", 44, 34,
         dict(**{"red-w": "8in", "sfu": "160pt", "title-x": "8.8in", "authors": "52pt",
                 "top": "4.35in", "body": "28pt", "body-lh": "36pt", "cap": "26pt",
                 "cap-lh": "34pt", "small": "20pt", "small-lh": "26pt", "head": "48pt",
                 "boxtitle": "36pt"}),
         f"""<div class="red"><div class="sfu">SFU</div></div>
<div class="title" data-fit="34.4" style="bottom:calc(100% - 2.36in); font-size:100pt">{W['PosterTitle']}</div>
<div class="authors">{W['PosterAuthors']}</div>
{qr("qr-line")}""",
         f"""<div class="body" style="grid-template-rows: 16.03in 12.25in; row-gap: .4in">
<div style="display:grid; grid-template-columns: 11in 30.4in; column-gap: 1in">
<div class="col" style="justify-content: space-between">
{box("The problem", W['TxtProblem'])}
{box("CANDI", W['TxtCandi'])}
{box("Key findings", W['TxtFindings'])}
</div>
<div class="col"><h2>{W['HeadStrip']}</h2>
<img src="{svg('ga_strip')}" alt="CANDI schematic workflow" style="width:100%">
<p class="cap wide">{W['CapStrip']}</p></div>
</div>
<div style="display:grid; grid-template-columns: 13in 28.4in; column-gap: 1in">
<div class="col"><h2>{W['HeadUtility']}</h2>
<div class="row">{fig('ga_A', 6.25, W['CapA'])}{fig('ga_B', 6.25, W['CapB'])}</div>
<div class="fill"></div>{refs()}</div>
<div class="col"><h2>{W['HeadEIC']}</h2>
<div class="row"><div class="text" style="width:7.6in"><p>{W['TxtEIC']}</p></div>
{fig('eic_leaderboard_landscape', 8.2, W['CapC'])}
{fig('eic_measures_landscape', 12.0, W['CapD'])}</div></div>
</div>
</div>""")

# --- portrait: poster_portrait.tex, 34 x 44 in -------------------------------
t1, t2 = "Self-supervised confidence-aware denoising", "imputation of genomic data"
assert f"{t1} {t2}" == W["PosterTitle"]
P = page("portrait", 34, 44,
         dict(**{"red-w": "7in", "sfu": "150pt", "title-x": "7.7in", "authors": "50pt",
                 "top": "4.45in", "body": "28pt", "body-lh": "36pt", "cap": "24pt",
                 "cap-lh": "31pt", "small": "19pt", "small-lh": "24pt", "head": "44pt",
                 "boxtitle": "34pt"}),
         f"""<div class="red"><div class="sfu">SFU</div></div>
<div class="title" data-fit="22" style="top:.55in; font-size:70pt">{t1}<br>{t2}</div>
<div class="authors">{W['PosterAuthors']}</div>
{qr("qr")}""",
         f"""<div class="body" style="grid-template-columns: 12.2in 19.4in; column-gap: .8in;
  grid-template-rows: 4.9in 16.85in 16in; row-gap: .5in">
<div class="row" style="grid-column: 1 / 3">
<div style="width:10.2in">{box("The problem", W['TxtProblem'], 4.9)}</div>
<div style="width:10.2in">{box("CANDI", W['TxtCandi'], 4.9)}</div>
<div style="width:10.2in">{box("Key findings", W['TxtFindings'], 4.9)}</div>
</div>
<div class="col" style="grid-column: 1 / 3"><h2>{W['HeadStrip']}</h2>
<img src="{svg('ga_strip')}" alt="CANDI schematic workflow" style="width:100%">
<p class="cap wide">{W['CapStrip']}</p></div>
<div class="col"><h2>{W['HeadUtility']}</h2>
<div class="row" style="align-items:center"><img src="{svg('ga_A')}" alt="" style="width:6.6in">
<p class="cap" style="width:5.2in; margin:0">{W['CapA']}</p></div>
<div class="row" style="align-items:center; margin-top:.4in"><img src="{svg('ga_B')}" alt="" style="width:6.6in">
<p class="cap" style="width:5.2in; margin:0">{W['CapB']}</p></div>
<div class="fill"></div>{refs()}</div>
<div class="col"><h2>{W['HeadEIC']}</h2>
<div class="row">{fig('eic_leaderboard_portrait', 7.2, W['CapC'])}
<div style="width:11.6in"><div class="text"><p>{W['TxtEIC']}</p></div>
<img src="{svg('eic_measures_portrait')}" alt="" style="width:100%; margin-top:.35in">
<p class="cap">{W['CapD']}</p></div></div></div>
</div>""")

for name, text in (("poster_landscape", L), ("poster_portrait", P)):
    f = OUT / f"{name}.html"
    f.write_text(text)
    print(f"wrote html/{f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
