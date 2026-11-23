# Log

Append-only. Written only by scripts/log.sh. Older entries rotate to log/.
- 2026-11-22T20:59Z longrun initialized (protected=cruxvault/INSTRUCTIONS/eic_entry, job_prefix=eicx_)
- 2026-11-22T21:01Z kickoff: ledger R1-R10, decisions D1-D4, two read agents gathering facts
- 2026-11-22T21:02Z glossary: reply sent with unregistered terms: log10
- 2026-11-22T21:03Z PI confirmed D2 (fairness rule) and ledger tiers
- 2026-11-22T21:04Z fact (Max s04): EIC scorer metrics9 = 9 measures on RAW QN-truth values at 25 bp, chr1-22+X minus blacklist, 10 chromosome-group bootstraps; truth cache Nibi /scratch/maxwl/epi_imputation/s04/cache/qn; readers bigwig:<dir> or candi:<root>/<exp>/<tag>/<chrom>.npz; no count reader
- 2026-11-22T21:04Z fact (Max s04): QN rule exact (apply_bp.py): per track peaks (own narrowPeak, M=1000)/background (M=50), stable sort ties by position, k=floor((r+1)(M-1)/N), value=R_assay[k], R = mean of 2019 training+validation per-level means; Zenodo SE record has bigwigs+peaks only, no reads
- 2026-11-22T21:04Z fact (Max s10): formula on predicted counts (macs2_expect, strict tail, truncation) vs direct p head: -0.019 [-0.062,+0.015] on 102 single-replicate targets; generic control ~ own control; never run on EIC blind; 25 bp overlap counts lose 0.10-0.23 (same reads) but only +0.002..+0.008 for imputation (independent reads)
- 2026-11-22T21:04Z fact (Max s04): CANDI+tcfloor 1st on MOESM4 (affine b0+b1*CANDI+b2*Average fitted on 133 training experiments; fair per D2) vs uncorrected CANDI 16th; Max's later plan drops tcfloor from challenge rows
- 2026-11-22T21:06Z spawned ladder probe agent (opus), 3 blind experiments, L1-L7
- 2026-11-23T06:15Z Ladder probe done (3 blind exps): QN exact, 1bp formula exact, overlap-count cost 0.017/0.255/0.021 nMSE, DNase full route exact with overlap peaks, ChIP reprocessing gap 0.013/0.127 unexplained. Recorded h90/f390 supports strong, q90 f391 moderate, 2 pending. Counters bumped h89 f389.
- 2026-11-23T06:18Z audit: first audit; fixed glossary status column + 2 rows (count route, conversion floor), R7 partial, D5 peak rule fixed a priori (D2), MOESM2 board added to Next, compute ask (82 CPU-h vs D4) added to STATE
