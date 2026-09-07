# Does searching over the initial noise, with the correction attached, give a crisper two-animal render?   ✅ support on the pre-registered bar · half the seeds by eye · verified 2026-09-06

**The claim**

With the rank-32 correction attached at λ 1.2, the best of eight nearby starting noises composes
on every held-out cat × dog seed, where the cached noise alone composes on six of eight. On four
seeds the kept render is a cleaner, crisper cat and dog than the correction's own render, and on
the seed the correction fails on, it is two animals.

On two seeds the pick is worse: the verifier's sharpness key, an edge count, prefers a drawn,
fur-textured image to a clean photograph. The bar written before the run reads the median edge
ratio and calls this support. The eye count is four gains, two holds, two losses, and the paper
says both.

Random search among the eight candidates, without the sharpness key, reaches the same eight of
eight, so the compose gain is having candidates, and the sharpness key only chooses which
composing candidate is shown.

## Table of contents

- [What we set out to see](#what-we-set-out-to-see)
- [What would have counted](#what-would-have-counted)
- [What was tried, in order](#what-was-tried-in-order)
- [1. Every seed composes with eight candidates, and the failing seed too](#1-every-seed-composes-with-eight-candidates-and-the-failing-seed-too)
- [2. The edge ratio says sharper on the median; the eye says four of eight](#2-the-edge-ratio-says-sharper-on-the-median-the-eye-says-four-of-eight)
- [3. On the composing pair, the correction is what does the harm](#3-on-the-composing-pair-the-correction-is-what-does-the-harm)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Where this came from](#where-this-came-from)
- [Depends on](#depends-on)
- [Still open](#still-open)

## What we set out to see

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-would-have-counted) ➡️

[The plain-PoE search](does-searching-over-the-initial-noise-compose.md) found no two-animal
noise near the failing ones. The corrected run composes already and its renders are soft, so the
same search was run with the adapter attached and the verifier changed: the compose count first,
so a one-animal candidate can never win, then sharpness, so among composing candidates the
crispest is kept. The goal in words was a render that snaps into two clean animals at full
sharpness from a seed whose corrected render is soft.

**The hypothesis.** Among eight nudges of the starting noise the corrected sampler produces a
two-animal render sharper than the cached noise's on most seeds, and the compose rate holds.

## What would have counted

Navigation: ⬅️ [Previous](#what-we-set-out-to-see) | 📋 [TOC](#table-of-contents) | [Next](#what-was-tried-in-order) ➡️

Support if, at either σ, the kept compose rate is within one seed of the adapter's own and the
median over seeds of the kept image's sharpness over the adapter render's is at least 1.10. Null if
compose holds and the median ratio is within 0.05 of 1 at both σ. Breaks if compose falls by more
than one seed. Sharpness is the variance of the second differences of the greyscale render (plan
07's Laplacian variance), compared only per seed because it counts edges. Constants
`MAX_COMPOSE_DROP`, `MIN_SHARPNESS_GAIN`, `NULL_SHARPNESS_BAND` in
`poe_repair/experiments/noise_search/adapter.py`; the question in
[the review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/12-noise-search-on-top-of-the-adapter.md).

**The result on the bar: support.** Kept 8 of 8 at both σ against 6 of 8; median ratio 1.47 at
σ 0.1 and 2.88 at σ 0.3.

## What was tried, in order

Navigation: ⬅️ [Previous](#what-would-have-counted) | 📋 [TOC](#table-of-contents) | [Next](#1-every-seed-composes-with-eight-candidates-and-the-failing-seed-too) ➡️

| When (2026-09-06) | What | Where it landed | Outcome |
|---|---|---|---|
| 02:38 | smoke on mscluster107 device 1: seed 9, N 2, σ 0.3, 10 steps, 512² | `outputs/interaction_term/noise_search/smoke_adapter_N2_s10_20260906-023829/` | 210 modules matched, 420 tensors loaded at step 30050; disabled adapter reproduces the pre-attach render to 0.0 |
| 02:42 to 04:31 | cat × dog on mscluster107 device 1, PID 57966, W&B `kqj26oiz` | `.../zo_adapter_N8_s50_20260906-024225/` | 8 seeds at 13 min each; verdict support |
| 02:43 to 03:52 | butterfly × meadow on mscluster109 device 1, PID 1860430, W&B `6oai0d99` | `.../zo_adapter_N8_s50_20260906-024309/` | 8 seeds at 8 min each; its detach check read 6.0 grey levels |
| 03:55 | the floor: plain PoE seed 9 twice in one process, no adapter, on the A6000 | `.../noise_search/floor/mscluster109/floor.json` | 0.0, so the 6.0 is the attach changing that card's numerics, within the same scene |
| 04:33 | both-ness on mscluster85, CPU, xformers disabled | `.../zo_adapter_N8_s50_20260906-024225/bothness.json` | the kept columns sit in the corrected band |

Every render: the seed's cached noise or a candidate at a recorded cosine to it, the rank-32
step-30050 adapter at λ 1.2 on every step as `eps_PoE(off) + 1.2 · (eps_PoE(on) − eps_PoE(off))`,
50 DDIM steps at `eta` 0, guidance 7.5, 1024². Mono and plain PoE were rendered before the adapter
was attached.

## 1. Every seed composes with eight candidates, and the failing seed too

Navigation: ⬅️ [Previous](#what-was-tried-in-order) | 📋 [TOC](#table-of-contents) | [Next](#2-the-edge-ratio-says-sharper-on-the-median-the-eye-says-four-of-eight) ➡️

![Cat × dog, seeds 9 to 16: Mono, plain PoE from the cached noise, the adapter's own render from the cached noise, then the best of 8 perturbed noises at σ 0.1 and 0.3 with the adapter attached, picked by count then sharpness; the detector's count on every tile](../../artifacts/results/is-the-gap-the-samplers-or-the-models/noise-search-adapter-cat-dog-sheet.png)
*Rows are seeds, the number on each tile is the animal count (green at 2 or more). What to notice:
the third column is two animals on six rows and one animal on seeds 11 and 14; the last two
columns are two animals on every row. Row 14 turns a child with one dog into a cat and a dog. Row
9 turns a clean photograph of a cat and a dog into drawn cats.*

**The number.** Compose rate over 8 seeds: plain PoE 0.0, the adapter's own render 0.75, the
kept image 1.0 at both σ. Of the 64 candidates per σ, 0.94 (σ 0.1) and 0.89 (σ 0.3) count two or
more. Under plan 11's count-then-confidence rule the kept rate is also 1.0 at both σ. From
`artifacts/results/is-the-gap-the-samplers-or-the-models/noise-search-adapter-cat-dog-summary.json`,
fields `pairs.a_cat__x__a_dog.compose_rate_by_column` and `secondary`.

## 2. The edge ratio says sharper on the median; the eye says four of eight

Navigation: ⬅️ [Previous](#1-every-seed-composes-with-eight-candidates-and-the-failing-seed-too) | 📋 [TOC](#table-of-contents) | [Next](#3-on-the-composing-pair-the-correction-is-what-does-the-harm) ➡️

**The number.** Median over seeds of the kept image's sharpness over the adapter render's: 1.47
at σ 0.1, 2.88 at σ 0.3. The kept image reaches the plain-PoE render's sharpness on 5 of 8 seeds
at both σ, and Mono's on 5 and 6 of 8. The adapter render itself is not below plain PoE on this
measure (median plain over adapter 0.81; Mono over adapter 1.16), so the measure does not read
the softness the corrected photographs show. Same summary file, fields
`secondary.sigma_*_median_ratio_kept_over_adapter_pivot`, `*_kept_reaches_plain_band`,
`*_kept_reaches_mono_band`, `median_ratio_plain_over_adapter`, `median_ratio_mono_over_adapter`.

**The eye read**, every kept tile against the adapter's own render, recorded in the review file:
cleaner cat and dog on seeds 10, 12, 14 and 16 at both σ; about equal on 13 and 15; worse on 9
(drawn cats, no dog, three times the edges) and 11 (a busier sketch with more heads, counts 4 and
6). Both-ness on the landing finding's axes agrees where it can: the kept means sit just under
the adapter render's (0.39 and 0.38 against 0.41), the four cleaner seeds at 0.38 to 0.47, seed 9
down to 0.29 and 0.17. From `noise-search-adapter-bothness.json`.

## 3. On the composing pair, the correction is what does the harm

Navigation: ⬅️ [Previous](#2-the-edge-ratio-says-sharper-on-the-median-the-eye-says-four-of-eight) | 📋 [TOC](#table-of-contents) | [Next](#what-this-cannot-tell-you) ➡️

![Butterfly × meadow, seeds 9 to 16: Mono, plain PoE, the adapter's own render, then the best of 8 at σ 0.1 and 0.3 with the adapter attached](../../artifacts/results/is-the-gap-the-samplers-or-the-models/noise-search-adapter-butterfly-meadow-sheet.png)
*What to notice: the second column is a photographic butterfly over flowers on every row; the
third column is an illustration on every row, with no butterfly on rows 13 and 15 and a faint one
on 12 and 16; the last two columns are illustrations with several crisp butterflies, and row 14 at
σ 0.3 is a printed pattern.*

**The number.** Count of the adapter's own render: 0 on seeds 12, 13, 15 and 16, so by the count
the correction loses the butterfly on 4 of 8 where plain PoE keeps it on 8 of 8. The kept tiles
count 2 or 3 on 8 of 8 at σ 0.3 and 7 of 8 at σ 0.1, always as several butterflies. Mean
sharpness: Mono 65, plain 69, the adapter render 503, the kept tiles higher. From
`noise-search-adapter-butterfly-meadow-summary.json`. A meadow is not an animal, so the count is
not this pair's compose read; the eye read is in the review file.

## What this cannot tell you

Navigation: ⬅️ [Previous](#3-on-the-composing-pair-the-correction-is-what-does-the-harm) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**The sharpness measure counts edges.** It prefers drawn and sketched styles to photographs, it
does not read the corrected photographs' softness, and the support on the bar is its number.
The four-of-eight is the eye's.

**The verifier is the judge.** The count passes a second head (plan 11) and the sharpness key
passes line work; the kept tiles on seeds 9 and 11 are what that costs.

**One round, eight candidates, two step sizes, one pair.** The paper iterates; this is the
smallest version, on the pair the adapter was held out on.

**The adapter render here is this sampler's.** It composes on 6 of 8 where the existing λ 1.2
renders compose on 7 of 8; the two DDIM code paths part late on seeds 11 and 14 (2 to 12 grey
levels, same scene).

**The disabled adapter is inert on one card and within-scene on the other.** Exact on the Quadro
RTX 8000, six grey levels on the A6000 with a 0.0 floor; the references were rendered before
the attach, so nothing here depends on it.

## Where this came from

Navigation: ⬅️ [Previous](#what-this-cannot-tell-you) | 📋 [TOC](#table-of-contents) | [Next](#depends-on) ➡️

| What | Source | Mark |
|---|---|---|
| The compose rates, ratios, band counts, verdict | `artifacts/results/is-the-gap-the-samplers-or-the-models/noise-search-adapter-cat-dog-summary.json`, `noise-search-adapter-verdict.json`, copied from the run dir 2026-09-06 | verified |
| The control pair's numbers | `.../noise-search-adapter-butterfly-meadow-summary.json` | verified |
| Both-ness per tile | `.../noise-search-adapter-bothness.json`, written 2026-09-06 on mscluster85 by `run_adapter.py --bothness-only` | verified |
| The detach checks and the floor | `.../noise-search-adapter-detach-checks.json` | verified |
| The two sheets | `.../noise-search-adapter-cat-dog-sheet.png`, `.../noise-search-adapter-butterfly-meadow-sheet.png` | verified |
| Every candidate and its scores | `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search/zo_adapter_N8_s50_20260906-024225/a_cat__x__a_dog/seed_<n>/sigma_<σ>/` and the `...-024309` run for the control pair | verified |
| The eye reads | the review file's `Written before the run, answered after` and `Asked after the result`, recorded as the tiles were opened | stated |
| The run | tasks **Run the smoke mode** and **Launch the full mode** in [the design](../../plans/06-is-the-gap-the-samplers-or-the-models/plans/baselines/12-noise-search-on-top-of-the-adapter.md); verdict in [its review file](../../plans/06-is-the-gap-the-samplers-or-the-models/review/12-noise-search-on-top-of-the-adapter.md); W&B `kqj26oiz` and `6oai0d99` | verified |
| Regenerate with | `ssh <node> 'GPU=<idx> nohup bash /home-mscluster/mmolefe/Playground/PhD/poe_repair_min/scripts/noise_search/run_noise_search.sh adapter-full --pairs <pair> > /datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search/logs/adapter_full_<pair>.log 2>&1 &'`, then `XFORMERS_DISABLED=1 <co3 python> -m poe_repair.experiments.noise_search.run_adapter --bothness-only --run-id <run> --device cpu` on the session node | verified |

## Depends on

Navigation: ⬅️ [Previous](#where-this-came-from) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

- The same search on plain PoE: [does searching over the initial noise compose](does-searching-over-the-initial-noise-compose.md)
- What the correction is: [the LoRA corrector](../../context/world/lora-corrector.md)
- What the compose scorer counts and what it cannot: [compose rate](../../context/world/compose-rate.md)
- The both-ness axes: [where each condition lands](../when-does-the-outcome-lock-in/where-does-each-condition-land.md)
- The paper: [the note](../../artifacts/notes/inference-time-scaling-as-a-search-over-noise/note.md)

## Still open

Navigation: ⬅️ [Previous](#depends-on) | 📋 [TOC](#table-of-contents)

- [ ] A verifier that reads clarity rather than edges (a DINOv2 distance to the seed's Mono
      render, or a sharpness measure that ignores line work) can re-pick from the candidates on
      disk with no render.
- [ ] The showcase can use the kept tiles on seeds 10, 12, 14 and 16 and the adapter's own render
      elsewhere; whether the paper does is a writing decision.
- [ ] The correction's effect on the composing pair (illustration style, lost butterfly on 4 of 8
      by count) is a finding about the adapter, not the search, and belongs beside the transfer
      results.

## Cross-references

- The mention of **Laplacian variance as a sharpness read**, and of its disagreement with the eye on this pair, in [can a corrector or a clean tail sharpen the adapter's renders](can-a-corrector-or-a-clean-tail-sharpen-the-adapters-renders.md) (relevance match): that finding measured the same band on the plain-PoE references and demoted it to a secondary read.
