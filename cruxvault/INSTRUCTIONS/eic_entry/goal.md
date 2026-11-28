---
title: EIC entry for miniCANDI, the PI's goal (verbatim)
date: 2026-11-22
task: t390
question: q90
---

# The PI's goal, verbatim from chat (2026-11-22)

> to make sure our "count to signal pvalue" formula is accurate, do you think it's possible for us to figure out the formula by trying to reproduce eic pvalue tracks from their bam? if we are sure that we have the "count to signal" function/pipeline figured out exactly, then comparing minicandi against challenge entrants will become fair and possible

> our goal for this session to figure out a fair and tested way to compare minicandi's outputs in count space against challenge entrants (in p-value space)
>
> you are allowed to use max's repo and results and files and findings to guide you throughout the process.
>
> the solution for this problem can be in finding the "pipeline" recipe that works. or we might need to modify how miniCANDI works (e.g. the resolution/binning/strand-aware fragment pileup, or 1 bp read starts, etc)
>
> you are allowed to explore different options
>
> this is an exploratory session done in a separate branch from the minicandi architecture optimization or the other session that's working on the new deviance-based metrics (eval set) on count space.
>
> your session's goal is on exploring how to make our mini-candi enter the EIC entrants ranking. fairly and correctly.
>
> so don't merge this new branch. this is exploratory

# Context the PI agreed to in the same conversation

> the above metric/eval set is defined for count data and its great.
> but if we want to compare minicandi against other EIC entrants, we have to compare on signal p value space. correct?

The agent answered yes (long-horizon decision D9: the EIC headline uses the challenge's own metrics against the organizers' corrected truth).
