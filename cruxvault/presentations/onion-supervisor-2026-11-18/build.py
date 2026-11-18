import re, json
src = open('onion-closeout-2026-11-18/index.html').read()
figs = dict(re.findall(r'"([a-z0-9_]+)":"(data:image/png;base64,[A-Za-z0-9+/=]+)"', src))
def fig(n, cap=""):
    return f'<figure><img src="{figs[n]}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'
def two(left, right): return f'<div class="two"><div>{left}</div><div>{right}</div></div>'
def box(title, items, cls=""):
    return f'<div class="box {cls}"><h3>{title}</h3><ul>' + ''.join(f'<li>{i}</li>' for i in items) + '</ul></div>'
S = []
def slide(kicker, title, body): S.append(f'<section class="slide"><p class="kicker">{kicker}</p><h2>{title}</h2>{body}</section>')

slide("peeling the onion &middot; 2026-11-09 to 2026-11-18", "Peeling the onion: what we did, and what we know now",
 '<p class="lead">The plan of 2026-11-09 had five parts. This deck follows them in order. For each part: what we did, and what we know now that we did not know before.</p>'
 '<table class="grid"><tr><th>Part of the plan</th><th>One-line result</th></tr>'
 '<tr><td>1. Sources of variation</td><td>Each layer now has a data view that isolates it, and a measured size.</td></tr>'
 '<tr><td>2. Objectives</td><td>Imputation, artifact correction and calibrated uncertainty are one training objective and three scores.</td></tr>'
 '<tr><td>3. Eval</td><td>A metric set with baselines and a selection rule; the variance decomposition; the answer to the resolution question.</td></tr>'
 '<tr><td>4. Data</td><td>The EIC data rebuilt with the 4th dimension, "measurement": 2,740 measurements with designed views and 11 covariates.</td></tr>'
 '<tr><td>5. miniCANDI</td><td>Three 1-million-parameter prototypes, 16 ablations, four design rules settled on all evaluation regions.</td></tr></table>'
 '<p class="note">Text drafted by the agent from the run records. A number with parentheses is a mean over 3 seeds with the seed range (maximum minus minimum).</p>')

slide("1 &middot; sources of variation", "Each source of variation got a data view that isolates it",
 '<table class="grid"><tr><th>Source in the plan</th><th>The view that isolates it</th><th>State</th></tr>'
 '<tr><td>Read sampling (the Poisson part)</td><td>Two pseudoreplicate halves of every track: a random split of its reads</td><td>built for every view</td></tr>'
 '<tr><td>Data processing</td><td>The same reads processed in designed ways: single-end, cropped to 36 and 50 bp, thinned to 1/2, 1/4, 1/8 depth, no duplicate removal, no mapping-quality filter</td><td>built</td></tr>'
 '<tr><td>Data generation</td><td>Biological replicates; 71 second experiments of the same cell type and assay (other lab and donor)</td><td>built; lab cannot be separated from donor</td></tr>'
 '<tr><td>Biological: cell type, position, assay</td><td>Pairs of tracks of one assay in two cell types; genome strata</td><td>built</td></tr>'
 '<tr><td>Biological: population</td><td>none</td><td>not addressed (the plan called it "nice to have")</td></tr></table>'
 + box("What we know now", ["Read sampling is by far the largest layer at 25 bp. The replicate layer and the cell-type layer are small beside it (slides 5 and 6).",
   "Lab and donor change together in ENCODE, so the pure lab effect has too few pairs. We report lab with donor as one effect.",
   "Per-library and per-lane units (sequencing and library layers) are not built yet: they are in the 722 rows of the next build."], "know"))

slide("2 &middot; objectives", "Three goals became one training objective and three scores",
 two(box("What we did", ["<b>Imputation:</b> predict a hidden track from the other tracks of the cell. Score: share of the cell-type-specific variance recovered above the read-sampling floor.",
   "<b>Artifact correction:</b> given covariates C of the input and C' of the wanted output, predict the same track as if measured under C'. Score: gap closed between plain depth scaling and the read-sampling limit, on view pairs never seen in training.",
   "<b>Calibrated uncertainty:</b> the model outputs a Negative Binomial per bin. Score: how close its probability integral transform is to uniform.",
   "One loss for all three: Negative Binomial likelihood on hidden tracks, on other views, and on hidden bins of the same view."])
   + box("What we know now", ["The three goals can be trained together in one small model and scored separately.",
   "No other loss beats the plain Negative Binomial likelihood on two or more models."], "know"),
   fig("objective", "The training objective: three kinds of target, one likelihood")))

slide("3 &middot; eval (a)", "A metric set with baselines and a fixed selection rule",
 two(box("What we did", ["One lead term per goal, each on a scale from a simple reference (0) to an ideal (1), plus a composite (0.5 imputation, 0.25 correction, 0.25 calibration).",
   "A second correction score, the <b>correction-only term</b>: how much of the covariate cost the model closes against its own same-view score.",
   "Training-free baselines scored on the same set: position average, marginal Negative Binomial, depth-only scaling, the read-half floor.",
   "A selection rule fixed before any run: adopt an arm only if the composite rises by more than twice the seed range and no goal falls by more than twice its range.",
   "256 evaluation regions; a pre-written check of whether 16 regions are enough."])
   + box("What we know now", ["The base models beat the position average on every term: composite 0.53 against 0.20.",
   "16 regions are <b>not</b> enough for the correction terms: the check failed its rule. Correction is now scored on all 256 regions.",
   "On the decided arms, 16 and 256 regions agree in sign in 48 of 50 cells."], "know"),
   fig("check", "The 16-versus-256 region check")))

slide("3 &middot; eval (b) &middot; variance decomposition", "Peeling the layers: read sampling dominates; replicate and cell-type layers are small",
 two(fig("ladder", "The replicate ladder on the real data (two-sided form)"),
   box("What we did", ["A ladder of pair values: two read halves, two biological replicates, two cell types. Each rung is the extra variance over the rung below.",
   "Assay-matched pairs, 95% bootstrap intervals over cell types, no pooling across assay classes."])
   + box("What we know now", ["Earlier pass at 25 bp: read sampling was 76% to 97% of the variance per assay; the replicate layer 0.1% to 6%.",
   "Broad histone marks: the sampling rung is 0.0463; the replicate rung adds 0.0005 (76 pairs).",
   "Cell-type rung: 0.00273 [0.0015, 0.0040] for broad marks (355 pairs), 0.00355 [0.0020, 0.0051] for narrow marks (445 pairs).",
   "A second experiment from another lab and donor adds no variance that these pairs can resolve beyond a biological replicate.",
   "Limit: DNase-seq rows are not interpretable yet (one track with 15,201 reads sets the common depth)."], "know")))

slide("3 &middot; eval (c) &middot; the resolution question", "Almost all variance below the 25 bp bin is read-sampling noise",
 two(fig("binvar", "Variance by bin size, from counts at 1 bp in the 256 evaluation regions"),
   box("What we did", ["Recounted 412 views at 1 bp and split the variance of each track by bin size (1, 5, 25, 50, 125 bp).",
   "Used biological replicates to separate reproducible signal from noise."])
   + box("What we know now", ["Below 25 bp lies 0.93 (broad marks) and 0.90 (narrow marks) of the base-by-base variance. Read sampling alone gives 0.96.",
   "Replicates reproduce only about 0.1% of that variance. A finer bin adds noise, not signal, for histone marks.",
   "From 25 to 125 bp, 7% (broad) and 13% (narrow) of the step is reproducible: a coarser bin starts to lose signal.",
   "So 25 bp is a sound choice on the data side. The model-side test (75 runs) is ready for the next session."], "know")))

slide("4 &middot; data (a) &middot; the 4th dimension", "The EIC data, rebuilt with the measurement dimension",
 two(box("What we did", ["Rebuilt every EIC experiment from its reads with the standard ENCODE pipeline of its assay, checked by accession.",
   "<b>2,740 measurements</b>: 2,328 training, 334 validation, plus the blind rows. A measurement is one cell type, one assay, one way of processing.",
   "Designed views of the same reads (single-end, crop 36 and 50, depth 1/2, 1/4, 1/8, no duplicate removal, no mapping-quality filter), each with two read halves.",
   "<b>11 covariates per measurement</b>: depth, run type, read length, duplicate removal, mapping-quality threshold, pipeline, pipeline version, lab, platform, antibody, donor.",
   "15,123 pairs; 18 view combinations are withheld from training and used only to score correction.",
   "Add-ons: 71 second experiments; QC side tables; a 1 bp region store."])
   + box("Not done yet", ["The blind tracks (722 rows) and 4 DNase-seq rows; the extended dataset beyond EIC."], "warn"),
   fig("data_state", "State of the data deliverables")))

slide("4 &middot; data (b)", "Do the counterfactual views help? Yes, for artifact correction",
 two(fig("views_conly", "Training on the base view only against all views"),
   box("What we know now (all 256 regions, 3 seeds)", ["Training on the base view only cuts the correction-only term by about half: 0.650 to 0.309 on the conv autoencoder, 0.616 to 0.301 on conv + attention. Both are more than twice the seed range; the U-Net shows the same.",
   "Imputation, calibration and the composite do not move.",
   "Scrambling the covariates in training also cuts the correction-only term on all three models (by 0.20 to 0.24). So the models do use the covariates.",
   "This is the direct evidence for the 4th dimension: the views and the covariates are what make correction work."], "know")))

slide("5 &middot; miniCANDI (a) &middot; architecture", "The prototype: counts, DNA and covariates in; a Negative Binomial out",
 two(box("Inputs and output (as in the plan)", ["<b>X</b>: counts of all assays of one cell type in a window of 1,024 bins of 25 bp (25.6 kb); some tracks hidden or missing.",
   "<b>S</b>: the DNA sequence of the window, through our own small DNA tower.",
   "<b>C and C'</b>: the covariates of the input experiments and of the wanted output experiments.",
   "<b>Output</b>: two Negative Binomial parameters per bin and assay; the mean carries the depth of C'.",
   "Covariates enter by feature-wise scaling and shifting (FiLM)."])
   + box("Three prototypes of about 1 million parameters", ["Conv autoencoder, conv + attention, U-Net. Each trained with 3 seeds on 80,000 windows."]),
   fig("arch", "Architecture comparison on all evaluation regions")
   + box("What we know now", ["No architecture wins: the three tie on the composite (0.53, 0.53, 0.49).",
   "A real trade-off: the U-Net imputes best (0.432 against 0.382 and 0.387); the conv models correct best (0.714 and 0.685 against 0.504)."], "know")))

slide("5 &middot; miniCANDI (b) &middot; ablations", "16 ablations on three models: few switches matter",
 two(fig("heat1", "Arm minus base, in units of the seed range: inputs, DNA and covariates"),
   box("What we did", ["39 single-switch arms on three models, 3 seeds each: ChIP control as input and output, average baseline input, two-step routing, fixed input depth, DNA on, off and fused in four ways, covariate embedding and injection, mask mode, losses, training views, nine count transforms."])
   + box("What we know now", ["Most switches stay inside the seed range. The simple recipe holds.",
   "The default count transform stays: two alternatives lower imputation on all three models.",
   "Our own DNA tower is as good as frozen Nucleotide Transformer embeddings.",
   "Signal p-value and peaks can be derived from the predicted Negative Binomial; the formula is built and reviewed, the scoring of all runs is next."], "know")))

slide("5 &middot; miniCANDI (c) &middot; settled on all regions", "Four design rules are decided; two wait for scores",
 two(fig("rescore", "The decisive arms rescored on all 256 regions"),
   '<table class="grid"><tr><th>Rule, written before the scores</th><th>Verdict</th></tr>'
   '<tr><td>Covariates are used for correction</td><td>settled, strong</td></tr>'
   '<tr><td>All counterfactual views stay in training</td><td>settled, strong</td></tr>'
   '<tr><td>Imputation loss weight 4</td><td>a candidate for conv + attention only; it costs correction on the conv autoencoder</td></tr>'
   '<tr><td>Fixed input depth</td><td>U-Net: correction +0.166, imputation -0.028; stays a candidate, enters no recipe</td></tr>'
   '<tr><td>Two-step routing; average baseline input</td><td>not decided: scores still running</td></tr></table>'
   + box("State", ["37 of 62 scorings on all regions are done; the rest run on Nibi.", "The design table has 31 choices, each with evidence, confidence, and the run that would settle it."])))

slide("summary", "What we know now that we did not know on 2026-11-09",
 '<table class="grid"><tr><th>Part</th><th>New knowledge</th></tr>'
 '<tr><td>Sources of variation</td><td>At 25 bp, read sampling is the dominant layer; replicate and cell-type layers are a few percent of it. Lab cannot be separated from donor in ENCODE.</td></tr>'
 '<tr><td>Objectives</td><td>One Negative Binomial likelihood trains imputation, correction and uncertainty together; no other loss does better.</td></tr>'
 '<tr><td>Eval</td><td>A tested metric set and selection rule. Correction needs all 256 regions. Below 25 bp there is almost no reproducible signal.</td></tr>'
 '<tr><td>Data</td><td>The measurement dimension works: views and covariates are what make correction possible; they do not change imputation.</td></tr>'
 '<tr><td>miniCANDI</td><td>Three small architectures tie; the U-Net imputes best and the conv models correct best. Four design rules are settled.</td></tr></table>'
 + two(box("Not done", ["Blind tracks and the EIC leaderboard entry.", "The extended dataset beyond EIC.", "The mix-and-match round that combines the helpful switches.", "Population-level variation."], "warn"),
       box("Next: scaling", ["Finish the scoring; run the mix-and-match round; choose the architecture.", "Then scale on the three axes of the plan: data, context, model size."])))

CSS = """
:root{--bg:#fbfaf7;--fg:#1d2330;--mut:#5b6472;--line:#d9d6cf;--card:#ffffff;--acc:#1f5fa8;--know:#e9f2ec;--knowb:#2f7d4f;--warn:#fbf0e3;--warnb:#b3661a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#14171c;--fg:#e7e9ee;--mut:#9aa3b2;--line:#2c323c;--card:#1b2027;--acc:#7fb2ee;--know:#18261e;--knowb:#5fbf86;--warn:#2a2117;--warnb:#e0a45c}}
:root[data-theme="dark"]{--bg:#14171c;--fg:#e7e9ee;--mut:#9aa3b2;--line:#2c323c;--card:#1b2027;--acc:#7fb2ee;--know:#18261e;--knowb:#5fbf86;--warn:#2a2117;--warnb:#e0a45c}
*{box-sizing:border-box}html{scroll-snap-type:y proximity}body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.slide{min-height:100vh;padding:28px 40px 36px;scroll-snap-align:start;border-bottom:1px solid var(--line);max-width:1500px;margin:0 auto}
.kicker{color:var(--acc);font-size:14px;letter-spacing:.06em;text-transform:uppercase;margin:0 0 6px}
h2{font-size:30px;line-height:1.2;margin:0 0 18px;text-wrap:balance}h3{font-size:15px;margin:0 0 6px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
.lead{font-size:20px;max-width:1000px}.note{color:var(--mut);font-size:14px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:22px;align-items:start}
.box{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:0 0 14px}
.box.know{background:var(--know);border-left:5px solid var(--knowb)}.box.warn{background:var(--warn);border-left:5px solid var(--warnb)}
ul{margin:0;padding-left:20px}li{margin:5px 0}
table.grid{border-collapse:collapse;width:100%;margin:0 0 16px;background:var(--card)}table.grid th,table.grid td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}table.grid th{font-size:14px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
figure{margin:0 0 14px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px}figure img{width:100%;height:auto;display:block}figcaption{color:#444;font-size:13px;padding:4px 4px 0}
.pos{position:fixed;right:14px;bottom:10px;color:var(--mut);font-size:13px}
@media (max-width:900px){.two{grid-template-columns:1fr}.slide{padding:20px 16px}h2{font-size:23px}}
"""
JS = """
const s=[...document.querySelectorAll('.slide')];let i=0;const p=document.querySelector('.pos');
function go(n){i=Math.max(0,Math.min(s.length-1,n));s[i].scrollIntoView({behavior:'auto'});p.textContent=(i+1)+' / '+s.length}
addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();go(i+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(i-1)}});
addEventListener('scroll',()=>{let b=0,d=1e9;s.forEach((x,k)=>{const t=Math.abs(x.getBoundingClientRect().top);if(t<d){d=t;b=k}});i=b;p.textContent=(i+1)+' / '+s.length});go(0);
"""
html = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Peeling the Onion</title><style>{CSS}</style></head><body>{"".join(S)}<div class="pos"></div><script>{JS}</script></body></html>'
open('onion-supervisor-2026-11-18/index.html','w').write(html)
print(len(S), 'slides', round(len(html)/1e6,2), 'MB')
