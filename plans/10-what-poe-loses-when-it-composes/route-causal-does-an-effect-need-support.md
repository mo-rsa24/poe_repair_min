# 🎯 Causal composition: does an effect need the subject's own pictures?
<!-- template: exempt. Reason: a skill-route runbook built to route-file-template, not a plan -->

> [!NOTE]
> **You will hold:** for six subjects (dog, camel, log, snowman, husky, camel in snow), how many
> training captions name them, how many of those pictures show the subject in the effect state by your
> own eye, what SDXL draws when asked directly, and what PoE draws · **Time:** ~8 h over three sittings
> · **Stop point:** the verdict on H-C1 and H-C2, judged against the pass marks in §7.5 of the
> [write-up](what-poe-loses-when-it-composes.md#75-pass-marks-set-before-the-renders)

Built to: route-file-template v1

## Table of contents

- [Quick context: where you are](#quick-context-where-you-are)
- [What you will achieve](#what-you-will-achieve)
- [Words this route uses](#words-this-route-uses)
- [Where everything lives](#where-everything-lives)
- [The route at a glance](#the-route-at-a-glance)
- [Schedule](#schedule)
- [Tasks](#tasks) (the AI runs these)
- [Instructions](#instructions) (you do these by hand)
- [Done already](#done-already)
- [Later](#later)
- [Why this chain](#why-this-chain)
- [Next step](#next-step)

## Quick context: where you are

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-you-will-achieve) ➡️

**Goal:** test whether PoE can show one concept acting on another (rain wetting a dog) only when the
subject's own training pictures already show it in that state. This is H-C1 in the
[write-up's hypotheses](what-poe-loses-when-it-composes.md#62-causal-hypotheses-tested-on-sdxl). Why
it matters, and why rain on a dog does not fuse the way a cat and a dog do, is in
[§5 of the write-up](what-poe-loses-when-it-composes.md#5-why-the-two-failures-differ).

**Objective (the whole chain):** for each subject, the support share $f_A$ counted by eye in the
training stand-in, and the effect count out of 5 seeds for each of six prompts.

**Outcome (the whole chain):**
- a census folder per subject with its caption list, its contact sheets and your calls;
- one summary table and one figure that a single command rebuilds;
- the render sheets for the new pairs;
- a verdict line per hypothesis.

**Found on disk:**
- The Re-LAION-2B-en mirror, 128 files, is at `/datasets/mmolefe/relaion2b-en/parquet/`.
- The camel and dog sheets are rendered and mostly called.
- The CLIP pilot ran and failed its own check.
- 90 camel renders from jobs 72616 and 72617 sit uncalled.

**The cluster is offline.** Steps 1 and 9 need nothing from it. Every other step waits for it.

## What you will achieve

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#words-this-route-uses) ➡️

| Evidence per subject | Step |
|---|---|
| caption hits: the subject alone, and with each effect family | 2 |
| the hit captions themselves, as a list you can read | 2 |
| contact sheets of the pictures behind a seeded sample of the hits, sorted into piles | 3 |
| your call on every picture: in the effect state, not, no subject, can't tell | 4 |
| $f_A$ with its 95% interval, the summary table and the census figure, rebuilt by one command | 5 |
| renders: the subject alone, B alone, joint, PoE, the named effect, PoE with the effect word | 6 |
| your call on every tile | 7 |
| the third concept: PoE of subject, carrier and effect word, on the seeds where two-concept PoE lost the effect | 8 |
| **the verdict on H-C1, H-C2 and H-C3** | **9** |

## Words this route uses

Navigation: ⬅️ [Previous](#what-you-will-achieve) | 📋 [TOC](#table-of-contents) | [Next](#where-everything-lives) ➡️

**Subject.** The thing that should change: dog, camel, log, snowman, husky.

**Carrier.** The thing that changes it: rain, fire, snow.

**Effect state.** How the subject looks once changed: a wet dog, a burning log, a melting snowman, a
snow-covered husky.

**Effect family.** The words a caption might use for the effect, listed per carrier below.

**Support share $f_A$.** Of the subject's pictures, the share in the effect state, counted by your eye.

**Pile.** A group of similar pictures, made by clustering CLIP image embeddings, so a contact sheet
shows one kind of picture at a time.

| Pair | Subject words (whole word, any case) | Effect families |
|---|---|---|
| dog + rain | dog, dogs, puppy | **wet:** wet, soaked, drenched, soaking, dripping, damp · **weather:** rain, raining, rainy, storm, downpour, puddle · **water:** swimming, bath, river, lake, splash, beach |
| camel + rain | camel, camels, dromedary | same three families |
| log + fire | log, logs, firewood | **fire:** burning, burnt, burned, charred, on fire, flames, ablaze, embers |
| snowman + fire | snowman, snowmen | **melt:** melting, melted, melt · **fire:** fire, flames, burning |
| husky + snow | husky, huskies | **snow:** snow, snowy, snowing, snow-covered, frost, frosted, blizzard |
| camel + snow | camel, camels, dromedary, bactrian | the same snow family |

## Where everything lives

Navigation: ⬅️ [Previous](#words-this-route-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-route-at-a-glance) ➡️

```
/datasets/mmolefe/relaion2b-en/
  parquet/                          the 128 Re-LAION-2B-en files (done)
  census/<pair>/                    one folder per pair, e.g. census/camel-rain/
    counts.json                     subject total, hits per effect family, files searched
    hits-<family>.parquet           every hit: url, caption, key, source file
    sample-<family>.parquet         the seeded draw that gets fetched
    images/<family>/<rank>.jpg      fetched pictures; fetch.json records dead links
    subject.json                    GroundingDINO: subject present or not, per picture
    piles/<family>/pile-NN/         CLIP clusters of the pictures with the subject in them

artifacts/results/does-an-effect-need-support/          local, pulled for looking (gitignored)
  <pair>/
    captions-<family>.md            the hit captions, a numbered list you can read
    piles/<family>/pile-NN.jpg      one contact sheet per pile, numbered tiles
    piles/<family>/pile-NN.md       tile number → caption → link, beside the sheet
    calls.json                      your calls: tile number, call
  census.md                         the summary table, rebuilt by step 5's command
  census.png                        the census figure, rebuilt by the same command

scripts/census/                     the code steps 2, 3 and 5 write (committed)
```

The camel and dog sheets already rendered stay where they are, in
`artifacts/results/does-a-camel-in-the-rain-come-out-wet/` and
`artifacts/results/does-a-dog-in-the-rain-come-out-wet/`.

## The route at a glance

Navigation: ⬅️ [Previous](#where-everything-lives) | 📋 [TOC](#table-of-contents) | [Next](#schedule) ➡️

▶ **Recommended:** `you: 1 → AI: 2 → AI: 3 ‖ AI: 6 → you: 4 (one pile per message) → AI: 5 → you: 7 → AI: 8 (you call its tiles inside it) → AI: 9`

**Two lanes, meeting twice.**

| Lane | Steps | Session | Writes |
|---|---|---|---|
| AI census | 2, 3, 5 | a census session | `scripts/census/`, `/datasets/.../census/`, `artifacts/results/does-an-effect-need-support/` |
| you look | 1, 4, 7 | a drip session, one item per message | `calls.json` files |
| AI renders | 6, 8, 9 | the walk session on [the dog-and-rain walk map](../../artifacts/drips/a-dog-and-rain/EXECUTE_WALK.md) | render sheets, the walk map, the verdict |

## Schedule

Navigation: ⬅️ [Previous](#the-route-at-a-glance) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

Three sittings. Every length is an estimate.

| Sitting | Run | Checkpoint |
|---|---|---|
| 1, ~2 h, cluster up | step 1, then step 2, then steps 3 and 6 launched | the counts table on screen, both jobs queued |
| 2, ~3 h | step 4 for camel + rain and dog + rain, then step 5 | $f_A$ for both animals, the first census figure |
| 3, ~3 h | step 4 for the other four pairs, step 7 | every census picture and render tile called |
| 4, ~1.5 h | step 8, then step 9 | the third-concept tiles called, the verdict lines |

## Tasks

*For the AI to run. Paste each prompt into the session named in its header.*

Navigation: ⬅️ [Previous](#schedule) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 2. 🔧 Count the hits and list the captions

You ~10 min · AI ~15 min in session, ~15 min on the cluster (estimate) · Opus, high · census session

**Objective:** subject totals and hits per effect family for all six pairs, over all 128 files.
**Outcome:** `counts.json` and `hits-<family>.parquet` per pair on `/datasets`; `captions-<family>.md` per pair locally.
**Serves:** the caption rows of every table.

<a href="../../artifacts/notes/what-poe-loses-when-it-composes/figures/fig-caption-evidence.png"><img src="figures/fig-caption-evidence.jpg" width="300" alt="The first census: whole-word counts for dog and camel captions, the breakdown of 25 wet-word dog captions, and the pilot funnel"></a>

```text
Background
This repo tests where product-of-experts (PoE) composition on SDXL fails. Work in the worktree
/home/molef/PhD/poe_repair_min-dog-and-rain on branch exec/dog-and-rain. The hypothesis (H-C1 in
plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md, section 6.2) is that PoE
can show a carrier's effect on a subject only when the subject's own training pictures show it in that
state. This step counts the training captions that might show it.

Objectives
1. Write scripts/census/count.py: for each pair in the route's "Words this route uses" table, count
   whole-word, case-ignoring caption hits for the subject alone and for the subject with each effect
   family, over all 128 parquet files.
2. Save every hit (url, caption, key, source file) as hits-<family>.parquet and the counts as
   counts.json under /datasets/mmolefe/relaion2b-en/census/<pair>/.
3. Pull the captions to artifacts/results/does-an-effect-need-support/<pair>/captions-<family>.md as a
   numbered list, first 500 hits per family in a seeded random order, with the total at the top.
4. Show me the counts table, one row per pair and family.

What to read
- plans/10-what-poe-loses-when-it-composes/route-causal-does-an-effect-need-support.md: the pair table.
- scripts/caption_wetness.py: the sample stage already counts camel and dog this way.
- artifacts/results/does-a-camel-in-the-rain-come-out-wet/caption-counts.md: the mirror and DuckDB.
- environment/overview.md and environment/storage.md: the co3 path, /datasets, the disk guard.

What is already known
- 2,097,693,557 captions in 128 files; 784,797 camel hits and 14,396,194 dog hits (caption-counts.md).
- A scan written to /datasets took 47 minutes; the working table on node-local disk is the fix
  (commit 1f97ae9, caption_wetness.py, sample stage).
- DuckDB 1.1.3 is at /datasets/mmolefe/relaion2b-en/bin/duckdb.
- 1,078 of 2,574 wet-word dog and camel hits in 4 files mention food (write-up section 7.2), so the
  caption list must be read, not only counted.

Rules
- One CPU job on bigbatch through scripts/cluster.sh run; commit before launching.
- Whole words only, so "rain" never matches "train" and "camel" never matches "Camelot".
- Do not fetch any image in this step.

What to hand back
- scripts/census/count.py, committed.
- /datasets/mmolefe/relaion2b-en/census/<pair>/counts.json and hits-<family>.parquet for six pairs.
- artifacts/results/does-an-effect-need-support/<pair>/captions-<family>.md for six pairs.
- The proof is the counts table printed in chat, with files searched = 128 on every row.

What you will see
- Along the way, the job id, then the counts table.
- At the end, a closing report: what was asked, the paths, the table, what is still open, and at
  most one question.

I judge it when the counts table is on screen and every row says 128 files.
```

- `added:` four more pairs (from your request for fire and snow) · captions saved as text (from your
  expected outcome: a list of training captions)
- `dropped:` nothing

### 3. 🔧 Build the folders for the eye check

You ~10 min · AI ~20 min in session, ~2 h on the cluster (estimate) · Opus, high · census session

**Objective:** for each pair and family, a seeded sample fetched, the pictures without the subject
removed, the rest sorted into piles, and a numbered contact sheet per pile.
**Outcome:** the `piles/` folders, on `/datasets` and pulled locally.
**Serves:** step 4.

<a href="../../artifacts/results/does-a-camel-in-the-rain-come-out-wet/pilot-ten-walk-pictures.jpg"><img src="figures/pilot-ten-walk-pictures.jpg" width="300" alt="The pilot's ten pictures: one real camel, boots, culottes, a coat, a recliner cover, two real dogs, an ornament, a book cover and a baseball player"></a>

```text
Background
This continues the census session after step 2. The pilot (scripts/caption_wetness.py) showed that
CLIP's animal check let through a boot, culottes, a book cover and a baseball player, and CLIP's wet
call was wrong on 7 of 7. This step keeps CLIP only for sorting, uses the repo's validated
GroundingDINO count to keep pictures that show the subject, and leaves every wet call to my eye.

Objectives
1. Per pair and family, draw a seeded sample of hits: 600 per family, or all hits if fewer.
2. Fetch the images (8 s timeout, dead links recorded, not retried) into images/<family>/.
3. Keep only pictures where GroundingDINO finds the subject at least once; record the rest.
4. Embed the kept pictures with CLIP ViT-bigG and cluster them into piles of about 30.
5. Draw one contact sheet per pile: numbered tiles, 6 across, with pile-NN.md beside it listing
   tile number, caption and link.
6. Pull artifacts/results/does-an-effect-need-support/<pair>/piles/ to the laptop.

What to read
- scripts/caption_wetness.py: the sample and fetch stages to reuse.
- artifacts/results/can-we-trust-the-compose-score/README.md (main checkout) and
  plans/02-can-we-trust-the-compose-rate/review/03-what-the-current-benchmarks-score.md: the validated
  GroundingDINO instance count and how it is run.
- plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md, section 7.2.

What is already known
- About 40% of links are dead: 63 of 100 camel and 59 of 100 dog pictures loaded (pilot).
- CLIP ViT-bigG laion2B-39B-b160k is cached on the cluster; open_clip 3.3.0 is in co3.
- Pictures are web photos, so the local copies stay under artifacts/results/, which is gitignored.

Rules
- One seed for every draw, written into the sample file, so a rerun gives the same pictures.
- Never call a picture wet or dry in code. Piles sort; my eye calls.
- Launch through scripts/cluster.sh run; GPU only for the GroundingDINO and CLIP passes.

What to hand back
- scripts/census/fetch.py, keep.py, piles.py and sheets.py, committed.
- The piles folders on /datasets and pulled locally, one contact sheet and one .md per pile.
- A printed table per pair: sampled, loaded, subject kept, piles.

What you will see
- Along the way, the job ids, then one contact sheet per pair as a first look.
- At the end, a closing report with the table, the paths, what is still open, and at most one
  question.

I judge it when I can open camel-rain/piles/wet/pile-01.jpg and read its captions beside it.
```

- `added:` GroundingDINO in place of CLIP's animal check (from the pilot's failure) · piles, so
  similar pictures sit together (from your ask to make the eye check easier)
- `dropped:` CLIP's wet call (why: wrong on 7 of 7)

### 5. 🔧 Rebuild the summary stats and figures with one command

You ~5 min · AI ~20 min · Sonnet, medium · census session

**Objective:** $f_A$ per pair with its 95% interval, the summary table and the census figure.
**Outcome:** `census.md` and `census.png`, and the command that rebuilds them.
**Serves:** the census rows of the verdict.

```text
Background
This continues the census session after my calls in step 4. Each calls.json holds tile number and
call (effect, not, no subject, can't tell).

Objectives
1. Write scripts/census/summarise.py: read every calls.json and counts.json, and compute per pair and
   family: pictures called, effect count, f_A with a Wilson 95% interval, and for a zero the rule-of-
   three bound (below 3 in n).
2. Write artifacts/results/does-an-effect-need-support/census.md: the table, plus up to five example
   captions per pair whose pictures I called in the effect state, each with its link.
3. Draw census.png: f_A per subject with its interval, log scale, one colour per carrier.
4. Print the one command that rebuilds both.

What to read
- The calls.json and counts.json files from steps 2 to 4.
- artifacts/notes/what-poe-loses-when-it-composes/make_figures.py: the house style for figures.

What is already known
- f_A is a share of pictures that show the subject, so "no subject" and "can't tell" are left out of
  the denominator and reported beside it.

Rules
- Copy calls as they are. Do not re-call anything.

What to hand back
- scripts/census/summarise.py, committed; census.md and census.png.
- The rebuild command, printed and pasted into census.md's first line.

What you will see
- At the end, the table and the figure on screen, and a closing report with at most one question.

I judge it when running the printed command redraws census.png and the table matches my calls.
```

- `added:` example captions with their pictures (from your expected outcome) · the Wilson interval
- `dropped:` nothing

### 6. 🔧 Render the new pairs and the named-effect test

You ~10 min · AI ~15 min in session, ~2 h on the cluster (estimate) · Opus, high · walk session

**Objective:** six columns on 5 seeds for log + fire, snowman + fire, husky + snow and camel + snow,
plus the walk map's second test (PoE with the effect word) for camel and dog.
**Outcome:** one sheet per pair with its sidecar.
**Serves:** the render rows of every table.

<a href="../../artifacts/notes/what-poe-loses-when-it-composes/figures/fig-causal-renders.png"><img src="figures/fig-causal-renders.jpg" width="300" alt="Camel and dog renders at seeds 1 and 4 across six prompts, with the reader's calls"></a>

```text
/drip-execute-plan --request artifacts/drips/a-dog-and-rain/EXECUTE_WALK.md

Background
This continues the walk on exec/dog-and-rain. The camel and dog sheets are rendered and called. The
walk map's second hypothesis test (PoE of "a camel" and "wet" against "a dog" and "wet") is written but
not rendered. This step renders it, and the same six columns for four new pairs.

Objectives
1. For camel and dog: PoE of "a <animal>" with "wet", and with "in the rain", plus "wet" and "in the
   rain" alone, seeds 1, 2, 4, 9, 10.
2. For each new pair, the first six columns of the write-up's section 7.4: subject alone, carrier
   alone, joint "a <subject> and <carrier>", PoE, the named effect asked directly, PoE with the effect
   word. The seventh, three-concept PoE, is step 8.
3. One sheet per pair, the columns in that order, the seeds down the page.

What to read
- The walk map, both hypothesis tests and "Out in the world".
- plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md, sections 7.3 to 7.5.
- scripts/concept_pair_sheet.py and .sbatch, and environment/hpc/nodes.md.

What is already known
- Every earlier tile: 50 DDIM steps, guidance 7.5, the cached a_cat__x__a_dog latent, an RTX 3090.
- --column single renders a prompt alone; --pairs renders joint and PoE (concept_pair_sheet.py).
- The named effects are "a burning log", "a melting snowman", "a snow-covered husky", "a snow-covered
  camel"; the effect words are "burning", "melting", "snowy", "snowy".
- Jobs 72616 and 72617 already rendered 90 camel pictures (SDXL on each sentence, PoE of "a camel"
  with each tail); none is called yet. Step 7 calls them.

Rules
- Write each new pair's pass marks into the walk map before its first render, copied from the
  write-up's section 7.5, and stop for my yes.
- Commit before each launch; launch through scripts/cluster.sh run with CLUSTER_DIR set to
  /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain
- Pin an RTX 3090 node and probe it first (nodes.md, bigbatch row).

What to hand back
- One sheet and sidecar per pair under artifacts/results/does-an-effect-need-support/<pair>/.
- The camel and dog named-effect sheet under artifacts/results/does-a-camel-in-the-rain-come-out-wet/.
- The walk map with the pass marks and the job ids.
- The proof is a tile check per sheet: every tile present, every sidecar at 50 steps and 7.5.

What you will see
- Along the way, the pass marks for my yes, then the job ids, then each sheet as it lands.
- At the end, a closing report with the sheets, the tile checks and at most one question.

I judge it when every sheet is on screen with a full tile check.
```

- `added:` four pairs (from your request) · the walk map's second test (written there 2026-10-22)
- `dropped:` SuperDiff for the new pairs (why: H-C1 is about plain PoE)

### 8. 🔧 The third concept: can PoE get the effect back?

You ~30 min · AI ~20 min in session, ~1 h on the cluster (estimate) · Opus, high · walk session

**Objective:** three-concept PoE (subject, carrier and effect word) on every seed where the effect was
reachable through language and two-concept PoE lost it, called by you tile by tile.
**Outcome:** the third-concept sheet per pair, its calls file, and the counted-seed table.
**Serves:** H-C3 ([§7.6 of the write-up](what-poe-loses-when-it-composes.md#76-the-third-concept-can-poe-get-the-effect-back)).

```text
/drip-execute-plan --request artifacts/drips/a-dog-and-rain/EXECUTE_WALK.md

Background
This continues the walk on exec/dog-and-rain, after every render tile has my call (step 7). The
question: where the joint prompt or the named effect shows an effect and two-concept PoE loses it,
does adding the effect word as a third expert bring it back? The write-up's section 7.6 predicts it
comes back for subjects whose own pictures show the effect (dog, log, husky) and not for those whose
pictures rarely do (camel, snowman). A camel or snowman coming back falsifies the support claim.

Objectives
1. Write the third-concept test into the walk map, copied from the write-up's section 7.6, and stop
   for my yes.
2. Pick the counted seeds from my calls only: a seed counts when the joint prompt or the named effect
   shows the effect on it and two-concept PoE does not. Show me the counted seeds per pair before
   any render.
3. Render three-concept PoE on the counted seeds: "a dog", "rain", "wet"; "a camel", "rain", "wet";
   "a log", "fire", "burning"; "a snowman", "fire", "melting"; "a husky", "snow", "snowy"; "a camel",
   "snow", "snowy".
4. Show me each tile beside its two-concept PoE tile and its joint or named-effect tile, one pair per
   message, and record my call: e (effect on the subject), n (not), h (half), x (subject lost).
5. Print the counted-seed table: pair, counted seeds, recovered, not recovered.

What to read
- The walk map, its hypothesis tests and "Out in the world".
- plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md, sections 5.4, 6.2
  (H-C3) and 7.6.
- The calls files beside every sheet from step 7.
- scripts/concept_pair_sheet.py, to extend PoE to a third condition.

What is already known
- The three-concept product adds one more guided difference: eps_empty + w(eps_subject - eps_empty)
  + w(eps_carrier - eps_empty) + w(eps_effect - eps_empty), at w = 7.5 (write-up section 7.6).
- Dog + rain: joint wet on 2 seeds plus 1 half, PoE wet on 2 (walk map, piece 1); which seeds those
  are comes from step 7's per-seed calls.
- Camel + rain: "a wet camel" wet on 4, PoE dry on 5, so up to 4 camel seeds can count.
- A half call counts as not shown, in the selection and in the result.

Rules
- Select seeds from my existing calls before rendering; never add a seed after the tiles land.
- If concept_pair_sheet.py takes two conditions, add a third and check it reproduces the two-concept
  PoE tile exactly when the third condition is the empty prompt; show me that check first.
- Never propose a call. I call every tile.
- Pin an RTX 3090, commit before launching, launch through scripts/cluster.sh run.

What to hand back
- The walk map with the test, the counted seeds and the job id.
- One sheet per pair: counted seeds down the page; joint or named effect, two-concept PoE and
  three-concept PoE across; its sidecar and calls file, under
  artifacts/results/does-an-effect-need-support/<pair>/third-concept.png.
- The counted-seed table, and the check line that the empty third condition reproduces two-concept
  PoE.

What you will see
- Along the way: the test for my yes, the counted seeds, the empty-condition check, then one pair per
  message for my calls.
- At the end, a closing report: the counted-seed table, which pairs came back, what is still open, and
  at most one question.

I judge it when every counted seed has my call and the table is on screen.
```

- `added:` the seed-selection rule (from your point that only pairs that lost an effect can test its
  recovery) · the empty-condition check, so a third expert is known to change nothing on its own
- `dropped:` uncounted seeds (why: they hold no lost effect to recover)

### 9. 🔧 Join the evidence and give the verdict

You ~10 min · AI ~20 min · Opus, high · walk session

**Objective:** one table per pair and a verdict on H-C1 and H-C2.
**Outcome:** the README of `artifacts/results/does-an-effect-need-support/`.
**Serves:** the verdict.

```text
Background
This closes the walk on exec/dog-and-rain. The census, the renders and every call are done.

Objectives
1. Build one table per pair: f_A by eye with its interval, then the effect count out of 5 for each
   of the seven columns.
2. Judge H-C1 and H-C2 against the pass marks in the write-up's section 7.5 and the walk map's tests,
   and H-C3 on the counted seeds only (section 7.6), and write the verdict lines.
3. Close the walk.

What to read
- artifacts/results/does-an-effect-need-support/census.md (step 5).
- Every calls file beside every sheet (steps 7 and 8).
- The walk map's tests and the write-up's section 7.5.

What is already known
- The census counts a public stand-in, not SDXL's data (write-up section 7.2).
- Every PoE miss on SDXL can still be the sampler (write-up section 11).

Rules
- Copy counts and calls as they are. Do not re-call any tile or picture.
- If a count is not reachable, say so in the table and say what the verdict can still claim.

What to hand back
- The README with one table per pair, the verdict lines with the numbers each rests on, and a
  thumbnail of each sheet.
- The walk map updated, and the /drip-retro command the close prints.

What you will see
- At the end, a closing report: the verdict lines, what is still open, and at most one question.

I judge it by eye, from the sheets, the census figure and the verdict lines.
```

- `added:` H-C2 (from the write-up) · the new pairs
- `dropped:` nothing

## Instructions

*For you to do by hand. Each prompt opens a drip: one item per message, you answer, it moves on.*

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#done-already) ➡️

### 1. 🧭 Tour what already exists

You ~20 min · Sonnet, medium · a fresh session

```text
Background
Work in /home/molef/PhD/poe_repair_min-dog-and-rain on branch exec/dog-and-rain. I am about to run a
census of training captions for six subject-and-carrier pairs. Before it starts I want to see, by
opening them myself, what already exists for camel and dog.

Objectives
1. Walk me through these, one per message, each with the command or click path to open it and what I
   should see:
   - the write-up's sections 1, 5 and 7: plans/10-what-poe-loses-when-it-composes/what-poe-loses-when-it-composes.md
   - caption-counts.md and the two pilot contact sheets in
     artifacts/results/does-a-camel-in-the-rain-come-out-wet/
   - the three sheets and their calls files (asked-directly, a-camel-and-rain, the dog sheet)
   - the mirror on the cluster: ls /datasets/mmolefe/relaion2b-en/ and one DuckDB count I type myself
2. Have me type the camel count myself:
   /datasets/mmolefe/relaion2b-en/bin/duckdb -c "SELECT count(*) FROM
   read_parquet('/datasets/mmolefe/relaion2b-en/parquet/*.parquet') WHERE
   regexp_matches(lower(caption), '\b(camel|camels|dromedary)\b');"
   and tell me the number to expect (784,797).

Rules
- One item per message, then stop for what I saw.
- Write nothing.

What you will see
- At the end, a closing report: what I opened, anything that did not match, at most one question.

I judge it when I have typed the camel count myself and seen 784,797.
```

### 4. ✍️ Call every picture, one pile at a time

You ~2.5 h over two sittings · Sonnet, medium · a drip session per pair

<a href="../../artifacts/results/does-a-camel-in-the-rain-come-out-wet/pilot-clip-called-wet.jpg"><img src="figures/pilot-clip-called-wet.jpg" width="300" alt="The seven pictures CLIP called wet in the pilot: camels on a wet street, a camel in grass, a DVD cover, a shoe, a puppy DVD cover, a button and an ink drawing; none shows a wet animal"></a>

```text
Background
Work in /home/molef/PhD/poe_repair_min-dog-and-rain. Step 3 built contact sheets of training pictures
whose captions name <pair: e.g. camel + rain>, sorted into piles. I call each picture by eye. The
pilot taught that the street can be wet while the camel is dry, so the call is about the subject.

Objectives
1. Show me one pile's contact sheet per message, from
   artifacts/results/does-an-effect-need-support/<pair>/piles/<family>/, with its pile-NN.md captions.
2. I reply with calls as tile number and letter: e for the effect state on the subject (wet fur,
   burning, melting, snow on the body), n for not, x for no subject, ? for can't tell.
3. Record each call in calls.json beside the sheets, and read back the running count.

Rules
- Never propose a call. I call every tile.
- One pile per message, then stop.
- If I ask to see one tile larger, open the full image.

What to hand back
- calls.json per pair: pile, tile, call, caption, link.

What you will see
- Along the way, one sheet per message and the running count of e, n, x and ?.
- At the end, a closing report: tiles called, the e count, three example captions with e, what is
  still open, and at most one question.

I judge it when every tile in every pile of the pair holds a call.
```

Order: camel + rain, dog + rain, then camel + snow, husky + snow, snowman + fire, log + fire.

### 7. ✍️ Call every render tile

You ~1 h · the walk session

<a href="../../artifacts/results/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain.png"><img src="figures/a-dog-and-rain.jpg" width="200" alt="The dog sheet: five seeds down the page, with columns for a dog alone, rain alone, the joint prompt, PoE and SuperDiff"></a>

```text
Background
This continues the walk on exec/dog-and-rain. Step 6's sheets have landed. Three sets of tiles are
still uncalled: the dog sheet's joint and PoE columns per seed (only totals exist), the 90 camel
renders from jobs 72616 and 72617, and every new sheet.

Objectives
1. Show me each sheet, its tiles numbered, one sheet per message.
2. Record my call per tile: e (effect on the subject), n (not), h (half), x (subject lost).
3. For the dog sheet, if my per-seed calls disagree with the walk map's totals (joint 2 wet plus 1
   half, PoE 2 wet), show me both and ask which stands.

Rules
- Never propose a call.

What to hand back
- A calls JSON beside every sheet: seed, column, my call.

What you will see
- At the end, the effect count per column per sheet, and at most one question.

I judge it when every tile holds a call.
```

## Done already

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#later) ➡️

| Step | Output | State |
|---|---|---|
| ~~Find a searchable stand-in~~ | Re-LAION-2B-en research-safe, all 128 files mirrored; DuckDB search; `caption-counts.md` "Dataset searched" | ✅ 2026-10-21 |
| ~~Render the four direct prompts~~ | `asked-directly.png`, 20 tiles, job 72389 | ✅ 2026-10-20 |
| ~~Render the camel sheet~~ | `a-camel-and-rain.png`, 25 tiles, job 72470 | ✅ 2026-10-20 |
| ~~Call the direct and camel sheets~~ | `asked-directly-calls.json`, `a-camel-and-rain-calls.json` | ✅ (the dog sheet's per-seed calls move to step 7) |
| ~~CLIP pilot, 100 captions per animal~~ | jobs 72598 and 72599, `/datasets/mmolefe/relaion2b-en/wetness/pilot/`; failed the 80% bar: 0 of 7 wet by Claude's eye | ✅ 2026-10-21; method replaced by steps 3 and 4 |

The prompts these steps ran are in this file's git history (`git log --follow` on this path).

## Later

Navigation: ⬅️ [Previous](#done-already) | 📋 [TOC](#table-of-contents) | [Next](#why-this-chain) ➡️

**SD3.5 as a second model.** The write-up's §5.7 predicts the joint prompt improves on causal pairs
and PoE does not. One sheet per pair on SD3.5 medium tests it.

**The image index.** `laion/relaion2B-en-research-safe-index` (about 0.5 TB) would allow searching by
picture rather than by caption, finding wet camels whose captions never name them.

## Why this chain

Navigation: ⬅️ [Previous](#later) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

<details><summary>Rationale and the time estimates</summary>

**Captions first, pictures second, eyes last.** A caption hit says a word appeared, and in the pilot most hits were products, idioms or the colour camel. Only a picture called by eye says the subject was in the effect state, so every number the verdict uses comes from your calls.

**Piles make the eye check fast.** Similar pictures together mean a pile of camel boots takes one glance and one "x" for the whole sheet.

**Six pairs, not two.** Dog against camel confounds wetness with how common each animal is (14.4 million against 0.78 million captions). Pairs across three carriers turn one comparison into a trend.

**Times.** Estimates: about 600 pictures per family, roughly half kept, about 5 seconds per tile.

</details>

## Next step

Navigation: ⬅️ [Previous](#why-this-chain) | 📋 [TOC](#table-of-contents)

**Start with:** instruction 1, the tour, today; it needs the cluster only for its last item. Then
step 2 once the cluster is back.
