# Does the pooled LoRA's held-out generalization gap close with more training pairs, or is something else costing the fit?

A cross-attention LoRA trained on 11 animal pairs composes 7 of 8 held-out cat-times-dog seeds at
its best checkpoints, then degrades with more training. The idea started as "more and cleaner
pairs close the gap, a tamer optimizer stops the decay," but two runs overturned the number the
plan was built on. The render-time cosine everyone was chasing (0.54 train, 0.40 held-out) turns
out to measure alignment with the pool-mean correction, not fit; the adapter's actual fit on
cached states is 0.97 train against 0.80 held-out, and its fit on states its own correction
actually visits at inference is 0.82 against 0.86 on cache, an exposure-bias cost of only 0.04.
The load-bearing claim is now split: the on-cache late-step gap (0.17) might still move with more
pairs, and that is the one question left open; on-policy training was demoted from the main fix to
a minor one.

## What is in here

Four files. `IDEA_MAP.md` carries the full `/drip-idea` walk: six claims, one split by two runs,
two dead ends (small pair-count tests too small to read), five outstanding checks, and four
completed runs. `routes/01-pressure-test-poe-failures.md` is a `/pressure-test` verdict against
the composition and LoRA-training literature, naming the three remaining failures (blend,
one-animal basin, late haze) and concluding the idea is promising but needs rework before more
data is cached. `run-01-fit-cosine-r8-030000.md` is the per-pair fit table on cached states (0.969
train mean, 0.800 held-out mean, gap widest in the late denoising steps). `run-02-on-policy-fit-r8-030000.md`
is the per-seed fit table on the adapter's own corrected trajectory (0.817 mean, against 0.860 on
cache, with state drift up to 0.86 by the last step).

## Where it came from and what judged it

**Held since 2026-09-05.** No plan file, master plan, or review file in this repo references the
idea yet, even though its two runs used real training checkpoints and cached evaluation cells
(`/datasets/mmolefe/poe_repair_min/outputs/showcase/fit_cosine_r8_030000/` and
`on_policy_fit_r8_030000/`) and its pressure-test route reached a verdict. Kept because the map's
own "Next step" leaves one decision open: whether to spend a 20-pair token-disjoint cache on
claim 1a's remaining 0.17 gap, or to work claim 3 (the late fidelity decay) first via a
zero-cost EMA test.
