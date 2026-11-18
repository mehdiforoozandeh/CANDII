shell = open('onion-closeout-2026-11-18/index.html').read()
a = shell.find('<section'); b = shell.rfind('</section>') + len('</section>')
S = []
def slide(sec, sub, h2, pts, notes, fig=None, table=None):
    cls = 'slide active' if not S else 'slide'
    f = f'<img class="figimg" data-fig="{fig}" alt="{sub}">' if fig else ''
    t = table or ''
    li = ''.join(f'<li>{p}</li>' for p in pts)
    S.append(f'<section class="{cls}"><div class="wrap"><div class="kicker"><span class="sec">{sec}</span>{sub}</div><h2>{h2}</h2>{f}{t}<ul class="pts">{li}</ul><aside class="notes"><p>{notes}</p></aside></div></section>')
def tbl(head, rows):
    return '<table class="sup"><tr>' + ''.join(f'<th>{h}</th>' for h in head) + '</tr>' + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows) + '</table>'

slide('peeling the onion', 'what we did, and what we know now',
 'Peeling the onion: five parts of the plan, and what each one gave us',
 ['For each part: <b>what we did</b>, and <b>what we know now</b> that we did not know on 2026-11-09.'],
 'This deck follows the order of the plan of November 9: sources of variation, objectives, eval, data, miniCANDI. Numbers in parentheses are seed ranges over 3 seeds.',
 table=tbl(['Part of the plan', 'One-line result'], [
  ['1. Sources of variation', 'Each layer has a data view that isolates it, and a measured size'],
  ['2. Objectives', 'Imputation, artifact correction and calibrated uncertainty: one training objective, three scores'],
  ['3. Eval', 'A metric set with baselines and a selection rule; the variance decomposition; the answer to the resolution question'],
  ['4. Data', 'EIC rebuilt with the 4th dimension, "measurement": 2,740 measurements, designed views, 11 covariates'],
  ['5. miniCANDI', 'Three 1-million-parameter prototypes, 16 ablations, four design rules settled']]))

slide('1 · sources of variation', 'a data view for each source',
 'Each source of variation got a data view that isolates it',
 ['<b>Know now:</b> read sampling is by far the largest layer at 25 bp; the replicate and cell-type layers are small beside it.',
  '<b>Know now:</b> lab and donor change together in ENCODE, so we report "lab with donor" as one effect.'],
 'Artifact variation has two kinds in the plan: data generation and data processing. Processing we can produce ourselves: the same reads processed in designed ways. Generation must come from ENCODE: replicates and second experiments. The library and lane layers are not built yet; they are part of the 722 rows of the next build. Population-level variation is not addressed.',
 table=tbl(['Source in the plan', 'The view that isolates it', 'State'], [
  ['Read sampling (Poisson part)', 'Two pseudoreplicate halves of every track', 'built'],
  ['Data processing', 'Same reads, designed views: single-end, crop 36 and 50 bp, depth 1/2, 1/4, 1/8, no duplicate removal, no mapping-quality filter', 'built'],
  ['Data generation', 'Biological replicates; 71 second experiments (other lab and donor)', 'built'],
  ['Biological: cell type, position, assay', 'Pairs of one assay in two cell types; genome strata', 'built'],
  ['Biological: population', 'none', 'not addressed']]))

slide('2 · objectives', 'one objective, three scores',
 'The three goals became one training objective and three scores',
 ['<b>Did:</b> one Negative Binomial likelihood on three kinds of target: a hidden track (imputation), another view of a track (artifact correction), hidden bins of the same view (denoising).',
  '<b>Did:</b> one score per goal: variance recovered above the read-sampling floor; gap closed on view pairs never seen in training; calibration of the predicted distribution.',
  '<b>Know now:</b> the three goals train together in one small model, and no other loss beats the plain likelihood on two or more models.'],
 'Correction means: given covariates C of the input and C-prime of the wanted output, predict the same track as if it was measured under C-prime. Uncertainty is aleatoric: the model predicts a distribution per bin, not a point value.',
 fig='objective')

slide('3 · eval', '(a) the metric set',
 'A metric set with baselines and a selection rule fixed before any run',
 ['<b>Did:</b> one lead term per goal on a scale from a simple reference (0) to an ideal (1), a composite, training-free baselines, and a rule: adopt an arm only beyond twice the seed range.',
  '<b>Know now:</b> the base models beat the position average on every term (composite 0.53 against 0.20).',
  '<b>Know now:</b> 16 regions are not enough for the correction terms: the pre-written check failed, so correction is scored on all 256 regions. Signs agree in 48 of 50 cells.'],
 'The composite is 0.5 imputation, 0.25 correction, 0.25 calibration. A second correction score, the correction-only term, measures how much of the covariate cost the model closes against its own same-view score. The baselines are the position average, a marginal Negative Binomial, depth-only scaling and the read-half floor.',
 fig='check')

slide('3 · eval', '(b) the variance decomposition',
 'Peeling the layers: read sampling dominates; the replicate and cell-type layers are small',
 ['<b>Did:</b> a ladder of pair values: two read halves, two biological replicates, two cell types; each rung is the extra variance over the rung below.',
  '<b>Know now:</b> broad marks: sampling rung 0.0463; replicate rung +0.0005; cell-type rung 0.00273 [0.0015, 0.0040]. Narrow marks: cell-type rung 0.00355 [0.0020, 0.0051].',
  '<b>Know now:</b> a second experiment from another lab and donor adds no variance that these pairs can resolve beyond a biological replicate.'],
 'An earlier pass at 25 bp gave the same picture: read sampling was 76 to 97 percent of the variance per assay, the replicate layer 0.1 to 6 percent. Pairs are assay-matched, intervals are bootstrap intervals over cell types. Limit: DNase-seq rows are not interpretable yet, because one track with 15,201 reads sets the common depth.',
 fig='ladder')

slide('3 · eval', '(c) the resolution question',
 'Almost all variance below the 25 bp bin is read-sampling noise',
 ['<b>Did:</b> recounted 412 views at 1 bp and split the variance of each track by bin size (1, 5, 25, 50, 125 bp).',
  '<b>Know now:</b> below 25 bp lies 0.93 (broad) and 0.90 (narrow) of the base-by-base variance; read sampling alone gives 0.96. Replicates reproduce about 0.1% of it.',
  '<b>Know now:</b> from 25 to 125 bp, 7% (broad) and 13% (narrow) of the step is reproducible: a coarser bin starts to lose signal.'],
 'This answers the interesting question of the plan on the data side: a finer bin than 25 bp adds noise, not signal, for histone marks. The model-side test, 75 training runs, is ready for the next session. DNase-seq has one track and no replicate pair here, so accessibility is not answered.',
 fig='binvar')

slide('4 · data', '(a) the 4th dimension: measurement',
 'The EIC data, rebuilt with the measurement dimension',
 ['<b>Did:</b> every EIC experiment rebuilt from its reads with the standard ENCODE pipeline, checked by accession: 2,740 measurements (2,328 training, 334 validation, plus blind rows).',
  '<b>Did:</b> designed views of the same reads, each with two read halves; 11 covariates per measurement; 15,123 pairs; 18 view combinations withheld from training to score correction.',
  '<b>Not done:</b> the blind tracks (722 rows), 4 DNase-seq rows, and the extended dataset beyond EIC.'],
 'A measurement is one cell type, one assay, one way of processing. The 11 covariates are depth, run type, read length, duplicate removal, mapping-quality threshold, pipeline, pipeline version, lab, platform, antibody and donor. Add-ons: 71 second experiments, QC side tables, a 1 bp region store.',
 fig='data_state')

slide('4 · data', '(b) do the counterfactual views help?',
 'The views and the covariates are what make artifact correction work',
 ['<b>Know now:</b> training on the base view only halves the correction-only term: 0.650 to 0.309 (conv autoencoder), 0.616 to 0.301 (conv + attention); the U-Net shows the same.',
  '<b>Know now:</b> scrambling the covariates in training cuts the same term on all three models, by 0.20 to 0.24.',
  '<b>Know now:</b> imputation, calibration and the composite do not move.'],
 'All of this is on all 256 evaluation regions with 3 seeds, and the falls are more than twice the seed range on the two conv models. This is the direct evidence for the fourth dimension: the model uses the covariates, and the designed views teach it the correction.',
 fig='views_conly')

slide('5 · miniCANDI', '(a) the architecture',
 'The prototype: counts, DNA and covariates in; a Negative Binomial out',
 ['<b>Inputs:</b> X, counts of all assays of one cell type in 1,024 bins of 25 bp; S, the DNA sequence through our own DNA tower; C and C-prime, covariates of the inputs and of the wanted outputs (injected by FiLM).',
  '<b>Output:</b> two Negative Binomial parameters per bin and assay. Three prototypes of about 1 million parameters: conv autoencoder, conv + attention, U-Net.',
  '<b>Know now:</b> no architecture wins on the composite (0.53, 0.53, 0.49). The U-Net imputes best (0.432 against 0.382 and 0.387); the conv models correct best (0.714 and 0.685 against 0.504).'],
 'This follows the specification in the plan. Hidden tracks are replaced by a learned token. The mean of the output carries the depth of C-prime. Each model was trained with 3 seeds on 80,000 windows. The choice of architecture for scaling is the first decision of the next session.',
 fig='arch')

slide('5 · miniCANDI', '(b) the ablations',
 '16 ablations on three models: few switches matter, the simple recipe holds',
 ['<b>Did:</b> 39 single-switch arms, 3 seeds each: ChIP control in and out, average baseline input, two-step routing, fixed input depth, DNA source and fusion, covariate embedding and injection, losses, training views, count transforms.',
  '<b>Know now:</b> most switches stay inside the seed range. The default count transform stays. Our own DNA tower is as good as frozen Nucleotide Transformer embeddings.',
  '<b>Know now:</b> signal p-value and peaks can be derived from the predicted Negative Binomial; the formula is built and reviewed, the scoring of all runs is next.'],
 'The figure shows arm minus base in units of the seed range: blue is up, red is down. A number appears only where the difference is beyond the range.',
 fig='heat1')

slide('5 · miniCANDI', '(c) settled on all regions',
 'Four design rules are decided on all evaluation regions; two wait for scores',
 ['<b>Settled, strong:</b> the covariates are used for correction; all counterfactual views stay in training.',
  '<b>Decided:</b> imputation weight 4 is a candidate for conv + attention only. Fixed input depth is a trade-off on the U-Net (correction +0.166, imputation -0.028); it enters no recipe yet.',
  '<b>Not decided:</b> two-step routing and the average baseline input; 37 of 62 scorings are done, the rest run now.'],
 'Each rule was written before its scores existed. The design table has 31 choices, each with its evidence, its confidence, and the run that would settle it.',
 fig='rescore')

slide('summary', 'what we know now',
 'What we know now that we did not know on 2026-11-09',
 ['<b>Not done:</b> blind tracks and the EIC leaderboard entry; the extended dataset; the mix-and-match round; population-level variation.',
  '<b>Next:</b> finish the scoring, combine the helpful switches, choose the architecture, then scale on data, context and model size.'],
 'One line per part of the plan. The prototype is ready to scale in the sense of the plan: the data, the objective and the evaluation are fixed, and the open design choices each have a ready run.',
 table=tbl(['Part', 'New knowledge'], [
  ['Sources of variation', 'At 25 bp read sampling is the dominant layer; replicate and cell-type layers are a few percent of it'],
  ['Objectives', 'One Negative Binomial likelihood trains imputation, correction and uncertainty together'],
  ['Eval', 'A tested metric set and rule; correction needs all 256 regions; below 25 bp there is almost no reproducible signal'],
  ['Data', 'The measurement dimension works: views and covariates make correction possible and do not change imputation'],
  ['miniCANDI', 'Three small architectures tie; U-Net imputes best, conv models correct best; four rules settled']]))

css = '<style>table.sup{border-collapse:collapse;width:100%;margin:10px 0 14px;font-size:.92em}table.sup th,table.sup td{border:1px solid rgba(128,128,128,.45);padding:6px 10px;text-align:left;vertical-align:top}table.sup th{font-size:.82em;text-transform:uppercase;letter-spacing:.04em;opacity:.75}</style>'
out = shell[:a] + '\n'.join(S) + shell[b:]
out = out.replace('</head>', css + '</head>', 1).replace('peeling the onion · close-out', 'peeling the onion · what we know now')
import re
out, n = re.subn(r"\(function\(\)\{\s*const L=.*?\}\)\(\);", "", out, count=1, flags=re.S)
assert n == 1 and "decTable" not in out
out = out.replace('<title>Onion Close-out</title>', '<title>Peeling the Onion</title>')
open('onion-supervisor-2026-11-18/index.html', 'w').write(out)
print(len(S), 'slides', round(len(out)/1e6, 2), 'MB', out.count('<section'))
