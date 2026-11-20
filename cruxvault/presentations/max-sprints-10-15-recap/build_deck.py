#!/usr/bin/env python3
"""Build index.html: a recap of Max's sprint 10 (covariate correction) and sprint 15
(paper planning). Source: github.com/mlibbrecht/01_2026-10-19_research at commit f5f6efb,
projects/03_2026-10-20_epi_imputation/sprints/{10_2026-11-11_covariate_correction,
15_2026-11-13_paper_planning}. Every number is Max's, copied from report_sections/00_summary.md,
04_counts.md, output/lead/budget_revised.tsv (copied beside this file) and paper_plan.md.
Nothing here is a CANDII vault number, so `crux deck --verify` does not apply.
Run: python3 build_deck.py   (chrome comes from ~/.claude/skills/prezit/assets/deck.html)
"""
import csv, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.expanduser("~/.claude/skills/prezit/assets/deck.html")

CAT = {  # category -> (label, colour)
    "unpredictable": ("Not predictable: a replicate does not reproduce it", "#8d8f98"),
    "covariate":     ("Explained by recorded covariates and depth", "#4a78b0"),
    "experiment":    ("Belongs to the individual experiment", "#d08a3c"),
    "reproducible":  ("Reachable by a better model", "#3f7d7a"),
}

def budget_svg():
    rows = [r for r in csv.DictReader(
        (l for l in open(os.path.join(HERE, "budget_revised.tsv")) if not l.startswith("#")),
        delimiter="\t") if r["row"] == "after"]
    tot = sum(float(r["value"]) for r in rows)
    x0, w, y, h = 20, 1060, 150, 110
    out, x = [], x0
    sums = {}
    for r in rows:
        sums.setdefault(r["category"], [x, 0.0]); sums[r["category"]][1] += float(r["value"])
        sw = float(r["value"]) / tot * w
        out.append(f'<rect x="{x:.1f}" y="{y}" width="{sw:.1f}" height="{h}" fill="{CAT[r["category"]][1]}" stroke="#faf9f7" stroke-width="2"/>')
        if sw > 60:
            out.append(f'<text class="val" fill="#fff" x="{x+sw/2:.1f}" y="{y+h/2+6}" text-anchor="middle">{float(r["value"]):.3f}</text>')
        x += sw
    stag = [40, 95, 40, 40]
    for i, (c, (cx, s)) in enumerate(sums.items()):
        sw = s / tot * w
        mid = cx + sw / 2
        ty = stag[i]
        out.append(f'<text class="barval" fill="{CAT[c][1]}" x="{mid:.1f}" y="{ty}" text-anchor="middle">{s/tot*100:.0f}%</text>')
        out.append(f'<text class="axlbl" x="{mid:.1f}" y="{ty+24}" text-anchor="middle">{s:.3f}</text>')
        out.append(f'<line class="axis" x1="{mid:.1f}" y1="{ty+32}" x2="{mid:.1f}" y2="{y-4}"/>')
    out.append(f'<text class="cap" x="{x0}" y="{y+h+28}">Normalized error of the previous CANDII recipe = {tot:.3f} (1 = no better than the track mean). One run, 282 targets.</text>')
    return f'<svg viewBox="0 0 1100 300" role="img">{"".join(out)}</svg>'

def forest(rows, lo, hi, xlabel, left=470):
    """rows: (label, sublabel, v, vlo, vhi, colour)"""
    W, rh, top = 1100, 78, 20
    H = top + rh * len(rows) + 70
    sx = lambda v: left + (v - lo) / (hi - lo) * (W - left - 60)
    o = []
    base = top + rh * len(rows) + 6
    t = lo
    step = (hi - lo) / 6
    ticks = [round(lo + i * step, 3) for i in range(7)]
    for t in ticks:
        o.append(f'<line class="grid" x1="{sx(t):.1f}" y1="{top}" x2="{sx(t):.1f}" y2="{base}"/>')
        o.append(f'<text class="tick" x="{sx(t):.1f}" y="{base+20}" text-anchor="middle">{t:+.2f}</text>')
    o.append(f'<line class="zero" x1="{sx(0):.1f}" y1="{top}" x2="{sx(0):.1f}" y2="{base}"/>')
    for i, (lab, sub, v, a, b, col) in enumerate(rows):
        cy = top + rh * i + rh / 2
        o.append(f'<text class="axlbl" x="{left-24}" y="{cy-2}" text-anchor="end">{lab}</text>')
        o.append(f'<text class="psub" x="{left-24}" y="{cy+18}" text-anchor="end">{sub}</text>')
        o.append(f'<line x1="{sx(a):.1f}" y1="{cy}" x2="{sx(b):.1f}" y2="{cy}" stroke="{col}" stroke-width="5" stroke-linecap="round"/>')
        o.append(f'<circle cx="{sx(v):.1f}" cy="{cy}" r="10" fill="{col}"/>')
        o.append(f'<text class="val" fill="{col}" x="{sx(b)+14:.1f}" y="{cy+6}">{v:+.3f} [{a:+.3f}, {b:+.3f}]</text>')
    o.append(f'<text class="ylab" x="{(left+W-60)/2:.1f}" y="{base+50}" text-anchor="middle">{xlabel}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img">{"".join(o)}</svg>'

G, O, B, T = "#8d8f98", "#d08a3c", "#4a78b0", "#3f7d7a"
control = forest([
    ("Control from the target experiment's own lab", "243 targets", -0.059, -0.073, -0.045, O),
    ("Control from another lab", "26 targets", 0.017, -0.011, 0.044, G),
], -0.09, 0.09, "Change in r_spec when the control is hidden from the model (95% interval)")
counts = forest([
    ("Counts and p-values from the same reads", "102 single-replicate targets", -0.019, -0.062, 0.015, G),
    ("Whole benchmark", "283 targets, most pooled from two replicates", 0.058, 0.024, 0.094, O),
], -0.12, 0.12, "Normalized error, count route minus direct p-value output (right = count route worse)")
ag2 = forest([
    ("AlphaGenome 2 lead over the CANDI four-run ensemble", "178 tracks AlphaGenome 2 never trained on", 0.021, 0.004, 0.037, B),
    ("What the same-lab control is worth to CANDI", "243 targets of our own benchmark", 0.059, 0.045, 0.073, O),
], -0.03, 0.15, "r_spec (95% interval)", left=560)

legend = "".join(f'<span class="item"><span class="sw" style="background:{c}"></span>{l}</span>' for l, c in CAT.values())

def slide(n, job, kicker, title, body, active=False):
    return f'''
  <!-- {n} · job: {job}
       source: Max's repo @ f5f6efb, sprints 10 and 15 (see build_deck.py)
       numbers: Max's, copied; not vault-addressed
       cut: whole slide -->
  <section class="slide{" active" if active else ""}">
    <div class="wrap">
      <div class="kicker">{kicker}</div>
      {title}
      {body}
    </div>
  </section>'''

S = []
S.append(slide(1, "title", "Recap · Max's sprints 10 and 15 · read 2026-11-15",
 "<h1>Covariate correction found no easy win.<br>The paper plan found two leaks.</h1>",
 '<p class="sub">Sprint 10 (Nov 11 to 13): how much experiment-specific variation is predictable. Sprint 15 (Nov 13 to 14): a paper plan with three independent reviews. Every number on these slides is Max\'s.</p><p class="hint">Navigate with <kbd>&rarr;</kbd> <kbd>&larr;</kbd> or <kbd>space</kbd></p>', True))

S.append(slide(2, "sprint 10 questions", "Sprint 10 · covariate correction · the questions",
 "<h2>Two experiments on the same cell type and assay differ. Max asked two things.</h2>",
 '''<div class="chain">
  <div class="node"><div class="lab">Question A</div><div class="txt">How much of the experiment-specific variation is predictable, even in principle?</div></div>
  <div class="arrow">&rarr;</div>
  <div class="node"><div class="lab">Question B</div><div class="txt">How do we impute in its presence? The candidate was a covariate artifact correction (CAC) module, i.e. remove an experiment's own patterns at the input and add them back at the output.</div></div>
  <div class="arrow">&rarr;</div>
  <div class="node now"><div class="lab">Outcome</div><div class="txt">A has an answer with ranges. B has a partial answer. No CAC module was built.</div></div>
 </div>
 <ul class="pts"><li><b>Replicate</b>: a second library of the same experiment. <b>Second experiment</b>: a separate ENCODE experiment on the same cell type and assay.</li></ul>'''))

S.append(slide(3, "error budget", "Sprint 10 · question A · result",
 "<h2>About 44% of the error is noise, 26% belongs to the experiment, 28% is reachable.</h2>",
 f'''<div class="fig wide">{budget_svg()}</div><div class="legend">{legend}</div>
 <ul class="pts"><li>Recorded covariates and depth explain only <b>2%</b>.</li>
 <li>The parts were measured on different subsets of targets, and the noise floor comes from replicates that were not matched. Max's report rounds the noise share to 45%.</li></ul>'''))

S.append(slide(4, "negative: covariates", "Sprint 10 · question B · the negative result",
 "<h2>No recorded covariate predicted the experiment's part.</h2>",
 '''<div class="cols2"><div class="blk"><div class="blab">What was tried</div>
 <ul class="pts"><li><b>Single-end against paired-end reads</b>: no detectable effect across 31 pairs of experiments. Effects up to about 0.08 in level error are not excluded.</li>
 <li><b>A corrector between real experiments</b> gained nothing from seeing their covariates.</li>
 <li><b>A planted-effect check</b>: the test detects a GC-shaped effect from 0.014 of a track's variance, so that null is real. For level it has no power below about 0.07.</li></ul></div>
 <div class="blk"><div class="blab">What it means for CAC</div>
 <div class="target"><div class="tlab">Observed experiment</div><div class="ttxt">A per-experiment level curve plus a GC-dependent term recovers up to about <b>0.08</b> of the 0.17 that belongs to the experiment.</div></div>
 <div class="target" style="margin-top:14px"><div class="tlab">Experiment never done</div><div class="ttxt">This is the imputation task. Max found <b>no way to predict</b> the experiment's part: a typical value recovers nothing of the GC term and about a tenth of level error.</div></div></div></div>'''))

S.append(slide(5, "control leak", "Sprint 10 · the ChIP-seq control",
 "<h2>The model uses the ChIP-seq control to predict the experiment, not the cell.</h2>",
 f'''<div class="fig wide">{control}</div>
 <ul class="pts"><li>The control helps only when it comes from the target experiment's own lab.</li>
 <li>Training with the control hidden at random roughly halved the loss when the control is missing. Four runs could not tell whether it costs anything when the control is present.</li>
 <li>Single-cell populations and most new datasets have no control, so the setting without the control is the realistic one.</li></ul>'''))

S.append(slide(6, "count route", "Sprint 10 · counts to p-values · bears on our count-only model",
 "<h2>The formula from counts to p-values is now exact. Predicted counts are not yet good enough.</h2>",
 f'''<div class="fig wide">{counts}</div>
 <ul class="pts"><li>Max's copy of the MACS2 formula (the ENCODE peak caller) had two details wrong. With them fixed it reproduces reprocessed tracks exactly from read positions.</li>
 <li>On pooled targets the count route loses: the model predicts one replicate, and its counts draw peaks too weak or too strong relative to background.</li>
 <li>On single-replicate targets a generic control (the mean of the training controls) works as well as the target's own.</li></ul>'''))

S.append(slide(7, "paper plan grid", "Sprint 15 · paper planning · the plan on one page",
 "<h2>Seven sections, one frozen model. The accuracy figure is the least settled.</h2>",
 '''<table class="dtable"><tr><th>Figure</th><th>Claim</th><th>Largest gap</th><th>Odds (Max's agent)</th></tr>
 <tr><td>1 · Model</td><td>One recipe, about 20 ablations</td><td>No single frozen network serves every section</td><td>High</td></tr>
 <tr><td>2 · Accuracy</td><td>Beats earlier imputers; open against AlphaGenome 2</td><td>No current model on the challenge split; two leaks; no retrained rival</td><td>Low to medium</td></tr>
 <tr><td>3 · Error breakdown</td><td>Most error is noise or the experiment's</td><td>Parts measured on different subsets</td><td>High</td></tr>
 <tr><td>4 · Confidence</td><td>90% intervals cover 90.7%</td><td>No calibration against replicates</td><td>High</td></tr>
 <tr><td>5 · Chromatin states</td><td>Annotation with a confidence per label</td><td>Nothing run yet</td><td>Pilot</td></tr>
 <tr><td>6 · Variant effects</td><td>Histone QTL Spearman 0.32 against 0.40 to 0.45</td><td>No held-out assay test</td><td>Pilot</td></tr>
 <tr><td>Single cell</td><td>A pseudobulk works as input</td><td>Old model only</td><td>Extended data</td></tr></table>
 <p class="note">Recommended scope: four figures at full depth, two pilots. Realistic date: early to mid January 2027. Proposed model freeze: Friday 2026-11-27.</p>'''))

S.append(slide(8, "AG2 leaks", "Sprint 15 · the referee-style review · AlphaGenome 2",
 "<h2>The control alone is worth about three times the gap to AlphaGenome 2.</h2>",
 f'''<div class="fig wide">{ag2}</div>
 <ul class="pts"><li><b>Leak 1, the control.</b> It is not recorded whether CANDI had the target's own control in that comparison. Headline numbers will give CANDI no control from the target's lab.</li>
 <li><b>Leak 2, the sequence encoder.</b> Borzoi trained on 132 of the 178 test tracks. The main figure will show the subset the encoder never saw.</li>
 <li><b>The fix:</b> one primary comparison written down before scoring, on tracks and chromosomes not yet scored, with a plain sentence for a loss.</li></ul>'''))

S.append(slide(9, "review good points", "Sprint 15 · what the three reviews changed",
 "<h2>The reviews removed claims that could not lose.</h2>",
 '''<ul class="pts lg cols">
 <li><b>Benchmark v3 is spent.</b> It was used to design the model. A fresh train, validation and test split is needed, and test is touched once at the end.</li>
 <li><b>Retrained rivals never drop.</b> Each rival must first reproduce its published numbers.</li>
 <li><b>Two challenge rows.</b> One with only what entrants had (no pretrained encoder, no blind controls), one with the encoder. No blending with the training average in the headline row.</li>
 <li><b>"Only CANDI predicts donor differences" was false.</b> Classic imputers also predict per sample. The claim now needs baselines and a second donor set.</li>
 <li><b>Model chosen on validation only.</b> Acceptance tests are written and dated before any candidate is trained.</li>
 <li><b>Fact check.</b> About 160 statements traced; 2 wrong, 16 had lost a qualifier.</li></ul>'''))

S.append(slide(10, "asks of Mehdi", "Sprint 15 · what the plan expects from us",
 "<h2>Three of our deliverables sit on Max's critical path.</h2>",
 '''<table class="dtable"><tr><th>Deliverable</th><th>Detail</th><th>Proposed date</th></tr>
 <tr><td>Count likelihood in CANDII</td><td>Ported and passing the freeze tests, or it enters a later version</td><td>Nov 25</td></tr>
 <tr><td>Challenge data view</td><td>For training on challenge data only; say whether the reprocessed set holds the blind tracks</td><td>Week of Nov 16</td></tr>
 <tr><td>Retrained rivals</td><td>ChromImpute, Avocado, eDICE on the 51 blind experiments and the fresh split</td><td>Week of Nov 30</td></tr>
 <tr><td>Pseudoreplicate halves</td><td>Depth-thinned halves and their sample list, for the denoising panel</td><td>Not set</td></tr>
 <tr><td>Variance layers</td><td>Results, one shared vocabulary, and which document carries which decomposition</td><td>Not set</td></tr></table>
 <div class="target" style="margin-top:22px"><div class="tlab">Conflict to settle with Max</div><div class="ttxt">The frozen model must keep a <b>signal output with a trained variance</b> and <b>variant-effect ability</b>. Our current prototype predicts counts only and rules population-level variation out of scope.</div></div>'''))

S.append(slide(11, "Max's three evaluations", "Max's message · 2026-11-15 · the paper's evaluations",
 "<h2>Max now wants three evaluations.</h2>",
 '''<table class="dtable"><tr><th>Evaluation</th><th>Split and units</th><th>Metrics</th><th>Compared with</th></tr>
 <tr><td>ENCODE Imputation Challenge</td><td>Challenge train, validation and test; log p-value signal; original and reprocessed</td><td>The 8 challenge metrics, without mse1imp</td><td>Challenge entrants, ChromImpute, eDICE. No pretrained sequence encoder.</td></tr>
 <tr><td>AlphaGenome 2</td><td>AlphaGenome 2's train and test split, as far as its API allows; fold change</td><td>Pearson, peak AUPRC, challenge metrics</td><td>AlphaGenome 2</td></tr>
 <tr><td>Our large dataset</td><td>A shared train, validation and test split, agreed once the new tracks exist</td><td>To agree</td><td>AlphaGenome 2 with its RNA encoder; Corgi+; CANDI with RNA; CANDI with RNA and ATAC</td></tr></table>
 <p class="note">Also from Max: no control track for single-cell or new datasets, so impute the control or predict signal directly. Test is touched only at the very end.</p>'''))

t = open(TEMPLATE).read()
head = t[:t.index('<div id="deck">')]
head = re.sub(r"<title>.*?</title>", "<title>Recap of Max's sprints 10 and 15</title>", head)
head = re.sub(r"(?s)<!--\s*=+\s*prezit DECK TEMPLATE.*?-->", "<!-- built by build_deck.py; chrome from the prezit template -->", head)
nav = t[t.index("/* ===========================================================================\n   NAVIGATION"):]
html = (head + '<div id="deck">' + "".join(S) + '\n</div>\n\n<div class="foot">Recap · Max\'s sprints 10 and 15</div>\n'
        '<div class="nav"><button id="prev" aria-label="previous">&larr;</button><button id="next" aria-label="next">&rarr;</button></div>\n'
        '<div class="count"><span id="cur"></span> / <span id="tot"></span></div>\n\n<script>\n' + nav)
open(os.path.join(HERE, "index.html"), "w").write(html)
print("slides:", len(S), "bytes:", len(html))
