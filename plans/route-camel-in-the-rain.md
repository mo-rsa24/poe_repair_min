# 🎯 Did SDXL see a wet camel, and does PoE only draw what it saw
<!-- template: exempt. Reason: a skill-route runbook built to route-file-template, not a plan -->

> [!NOTE]
> **You will hold:** one table per animal, with what the captions show, what SDXL draws when asked
> directly, and what [PoE](../context/world/poe-composition.md#what-poe-composition-is) draws · **Time:** ~3 h · **Stop point:** the verdict on the hypothesis test
> written in prompt 4, inside one 3-hour sitting today

Built to: route-file-template v1

## Table of contents

- [Quick context: where you are](#quick-context-where-you-are)
- [What you will achieve](#what-you-will-achieve)
- [The route at a glance](#the-route-at-a-glance)
- [The two lanes](#the-two-lanes)
- [Schedule](#schedule)
- [Tasks](#tasks) (the AI runs these)
- [Instructions](#instructions) (you do these by hand)
- [Quizzes and feedback](#quizzes-and-feedback)
- [Later](#later)
- [Why this chain](#why-this-chain)
- [References](#references)
- [Next step](#next-step)

## Quick context: where you are

Navigation: 📋 [TOC](#table-of-contents) | [Next](#what-you-will-achieve) ➡️

**Goal:** test the hypothesis that PoE can only render an overlap of two concepts that SDXL saw in
training.

**Objective (the whole chain):** for the dog and the camel, the number of verified wet images in
the caption search, the wet count out of 5 seeds when SDXL is asked directly, and the wet count
out of 5 under PoE.

**Outcome (the whole chain):** `artifacts/results/does-a-camel-in-the-rain-come-out-wet/` holding
`caption-counts.md`, the direct-prompt sheet, the camel-and-rain sheet, a calls file per sheet,
and a README with one table per animal and the verdict.

**Found on disk:** the walk to resume is `artifacts/drips/a-dog-and-rain/EXECUTE_WALK.md`. The dog
sheet exists, and the walk map records only its totals, not a call per seed.

**Plan weight:** none up front, because the walk adds its steps to the existing map.

**Who you are for this route:** a measurement. Nothing new to learn at depth.

## What you will achieve

Navigation: ⬅️ [Previous](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-route-at-a-glance) ➡️

| Evidence | Dog | Camel |
|---|---|---|
| Verified wet images in the captions, per word family | prompts 1 to 3 | prompts 1 to 3 |
| Wet out of 5 for "a wet dog" or "a wet camel", and "a dog in the rain" or "a camel in the rain" | prompt 4 | prompt 4 |
| Wet out of 5 for the joint prompt and PoE | dog sheet, called in prompt 6 | prompt 5 |
| **Verdict** | | **prompt 7, on the hypothesis test** |

## The route at a glance

Navigation: ⬅️ [Previous](#what-you-will-achieve) | 📋 [TOC](#table-of-contents) | [Next](#the-two-lanes) ➡️

▶ **Recommended:** `AI: prompts 4 and 5 launch ‖ you: prompts 1 → 2 → 3 → you: prompt 6 → AI:
prompt 7` (proposed: no curated flow fits)

<details><summary>Alternatives</summary>

Skip the caption search and judge from the direct prompts alone: better when the stand-in cannot
be searched at all, which prompt 1 finds out.

</details>

## The two lanes

Navigation: ⬅️ [Previous](#the-route-at-a-glance) | 📋 [TOC](#table-of-contents) | [Next](#schedule) ➡️

| Lane | Prompts | Session | Writes |
|---|---|---|---|
| AI renders | 4, 5, 7 | the walk session on `EXECUTE_WALK.md` | the walk map, sheets, README |
| You search | 1, 2, 3 | a separate session, pasted in order | `caption-counts.md` only |
| You call | 6 | the walk session | a calls file per sheet |

The lanes meet at prompt 7.

## Schedule

Navigation: ⬅️ [Previous](#the-two-lanes) | 📋 [TOC](#table-of-contents) | [Next](#tasks) ➡️

One 3-hour window today. Times are from the start of the window, and every length is an estimate.

| Block | Run | Checkpoint at the end of the block |
|---|---|---|
| 0:00 to 0:20 | prompts 4 and 5 in the walk session | hypothesis test confirmed, both jobs queued |
| 0:20 to 1:50 | prompts 1, 2 and 3 in the search session ‖ background: renders | eight counts and the checked images in `caption-counts.md` |
| 1:50 to 2:35 | prompt 6 in the walk session | every tile called |
| 2:35 to 3:00 | prompt 7 in the walk session | the tables, the verdict, the closing report |

If prompt 3 runs long, check 5 images per family instead of 10. That halves the verified counts'
size, so the README says which size was used.

## Tasks

*For the AI to run. Paste each prompt into the walk session.*

Navigation: ⬅️ [Previous](#schedule) | 📋 [TOC](#table-of-contents) | [Next](#instructions) ➡️

### 4. 🔧 Write the hypothesis test, then render the direct prompts

You ~10 min · AI ~10 min in session, ~30 min on the cluster (estimate) · Opus, high

**Objective:** 20 tiles, four prompts on 5 seeds, and a hypothesis test you confirmed before any render.
**Outcome:** the direct-prompt sheet with its sidecar.
**Serves:** the "asked directly" row of both tables.

<a href="../artifacts/results/does-a-camel-in-the-rain-come-out-wet/asked-directly.png"><img src="../artifacts/results/does-a-camel-in-the-rain-come-out-wet/asked-directly.png" width="140" alt="The asked-directly sheet: five seeds down the page, with columns for a wet camel, a camel in the rain, a wet dog and a dog in the rain, each prompt rendered alone"></a>

```text
/drip-execute-plan --request artifacts/drips/a-dog-and-rain/EXECUTE_WALK.md

Background
This repo tests where product-of-experts (PoE) composition on SDXL fails. Work in the worktree
/home/molef/PhD/poe_repair_min-dog-and-rain on branch exec/dog-and-rain. The walk map above asked
whether SDXL draws a wet dog for "a dog" and "rain". The hypothesis now under test is that PoE can
only draw an overlap SDXL saw in training. This prompt asks SDXL directly for a wet camel and a
wet dog, so the PoE result has something to be compared with.

Objectives
1. Write the hypothesis test in the walk map, then stop until I confirm or change it.
2. Render "a wet camel", "a camel in the rain", "a wet dog" and "a dog in the rain" alone.
3. Build one sheet from those 20 tiles: the four prompts as columns, the seeds down the page.

What to read
- artifacts/drips/a-dog-and-rain/EXECUTE_WALK.md: the walk's decisions, dog test, jobs and repairs.
- artifacts/results/does-a-dog-in-the-rain-come-out-wet/README.md: how the dog sheet was made.
- scripts/concept_pair_sheet.py and scripts/concept_pair_sheet.sbatch: renderer and launcher.
- environment/overview.md and environment/hpc/nodes.md: partitions, GPU models, dead nodes.

What is already known
- The dog sheet used seeds 1, 2, 4, 9 and 10, 50 DDIM steps and guidance 7.5 (results README).
- Each row starts from the cached a_cat__x__a_dog latent at that seed (results README).
- The "single" column renders each half of a pair alone (concept_pair_sheet.py, render).
- So --column single --pairs "a wet camel|a camel in the rain" "a wet dog|a dog in the rain"
  renders all 20 tiles with no new sampler code.
- The --sheet mode expects the pair columns, so the direct sheet needs a small new layout.
- Render skips existing tiles, so nothing already rendered is rewritten.
- The GPU model flips seed 10's final picture (walk map repairs).
- Job 65969's log names the card the dog sheet ran on: pairsheet-65969.out (walk map).
- Idle bigbatch nodes are often dead cards, so probe one and pin it (nodes.md, bigbatch row).
- Tiles land under /datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet (sbatch).
- scripts/cluster.sh pull <path> copies tiles back to the laptop (cluster.sh line 12).
- artifacts/results/ is gitignored (.gitignore line 40), so sheets and READMEs stay local.
- The dog sheet's sidecar lists tile paths in a scratchpad that no longer exists.

Rules
- Use this branch and this map. Do not start a new walk.
- Do not touch the dog sheet, its tiles or its sidecar.
- Before any render, write this hypothesis test in the walk map, and stop for my yes.
- The test: for each animal, PoE draws wet on no more seeds than "a <animal> in the rain" alone.
- It also needs the animal with fewer verified wet images in the captions to come out wet
  on fewer PoE seeds.
- A tie between the two animals' PoE counts is inconclusive.
- Pin the same GPU model the dog sheet used, read from the 65969 log.
- Commit before each launch, because scripts/cluster.sh refuses uncommitted tracked changes.
- Launch from the laptop through scripts/cluster.sh run, with CLUSTER_DIR set to
  /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain
- Never write caption-counts.md, which the search session owns.

What to hand back
- 20 tiles on the cluster, each a PNG with its own JSON sidecar, one folder per prompt:
  /datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/single/a_wet_camel/seed01.png
  and the same for a_camel_in_the_rain, a_wet_dog and a_dog_in_the_rain.
- Each folder holds seed01, seed02, seed04, seed09 and seed10.
- Local copies of those 20 tiles and sidecars, pulled with scripts/cluster.sh pull, in
  artifacts/results/does-a-camel-in-the-rain-come-out-wet/tiles/
- The sheet built from them:
  artifacts/results/does-a-camel-in-the-rain-come-out-wet/asked-directly.png
- Beside it, asked-directly.json listing every tile's cluster path and its local copy.
- The walk map, with the hypothesis test and this job's id.
- The proof is a printed line: 20 of 20 tiles, each sidecar at 50 steps, guidance 7.5 and the
  a_cat__x__a_dog latent.

What you will see
- Along the way, the hypothesis test for my confirm, then the job id once it is queued.
- At the end, a closing report: what was asked, the sheet's path, the tile check line, what is
  still open, and at most one question.
- The expected outcome is 20 tiles on one sheet, with no calls yet.

I judge it when the hypothesis test is in the walk map and the tile check prints 20 of 20.
```

- `added:` the "single" column as the direct renderer (from `concept_pair_sheet.py`) · the GPU
  model pin (from walk map repairs and `nodes.md`) · the hypothesis test (from your message)
- `dropped:` the CLIP scorer, the three-concept PoE, the camel-and-wet and dog-and-wet sheets
  (why: your objective does not need them)

### 5. 🔧 Render the camel-and-rain sheet

You ~5 min · AI ~5 min in session, ~60 min on the cluster (estimate) · Opus, high

**Objective:** 25 tiles in the dog sheet's shape.
**Outcome:** `a-camel-and-rain.png` with its sidecar.
**Serves:** the joint prompt and PoE rows of the camel's table.

```text
Background
This continues the walk on exec/dog-and-rain, right after prompt 4's job was queued. The
hypothesis test is already confirmed in the walk map. This prompt renders the camel's version
of the dog sheet.

Objectives
1. Render "a camel" and "rain" in the dog sheet's shape: each alone, the joint prompt, PoE and
   SuperDiff, seeds 1, 2, 4, 9 and 10.
2. Build the sheet on the laptop once the tiles are pulled.

What to read
- The walk map, for the confirmed hypothesis test and prompt 4's job.
- scripts/concept_pair_sheet.sbatch: its header shows the two launch lines this needs.

What is already known
- The joint template is "{a} and {b}", so the joint prompt reads "a camel and rain" (config.py
  line 53).
- PoE is run_cfg_poe in poe_repair/methods/_sampling.py, with no adapter attached.
- SuperDiff draws its own noise from the seed and runs 200 steps, since 50 shatters (walk map).
- SuperDiff ran as its own job on biggpu, mscluster110 (walk map, job 65970).
- The other four columns use the same GPU model pin as prompt 4.

Rules
- Do not render anything until prompt 4's hypothesis test is confirmed.
- Commit before each launch, and launch through scripts/cluster.sh run with the same CLUSTER_DIR.
- Show me the sheet when it lands, with its tile check line.

What to hand back
- 25 tiles on the cluster, each a PNG with its own JSON sidecar, under
  /datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet
- "a camel" alone goes to single/a_camel/, and the joint, PoE and SuperDiff tiles go to
  a_camel__x__rain/seedNN/ as mono.png, poe.png and superdiff_200.png.
- "rain" alone already exists in single/rain/ from the dog sheet, and is reused as it is.
- Local copies of all 25 tiles and sidecars, pulled with scripts/cluster.sh pull, in
  artifacts/results/does-a-camel-in-the-rain-come-out-wet/tiles/
- The sheet built from them:
  artifacts/results/does-a-camel-in-the-rain-come-out-wet/a-camel-and-rain.png
- Beside it, a-camel-and-rain.json listing every tile's cluster path and its local copy.
- Both job ids in the walk map.
- The proof is a printed line: 25 of 25 tiles, every sidecar at 50 steps, guidance 7.5 and the
  a_cat__x__a_dog latent, with SuperDiff at 200 steps and its own noise.

What you will see
- Along the way, the two job ids, then the sheet and its check line.
- At the end, a closing report: what was asked, the sheet's path, the check line, what is still
  open, and at most one question.

I judge it when the sheet is on screen and the check prints 25 of 25.
```

- `added:` nothing beyond your brief
- `dropped:` nothing

### 7. 🔧 Join the evidence and give the verdict

You ~10 min · AI ~15 min · Opus, high

**Objective:** one table per animal and a verdict on the hypothesis test.
**Outcome:** the camel folder's README.
**Serves:** the verdict row.

<a href="../artifacts/results/does-a-camel-in-the-rain-come-out-wet/asked-directly.png"><img src="../artifacts/results/does-a-camel-in-the-rain-come-out-wet/asked-directly.png" width="140" alt="The asked-directly sheet: five seeds down the page, with columns for a wet camel, a camel in the rain, a wet dog and a dog in the rain, each prompt rendered alone"></a>

```text
Background
This closes the walk on exec/dog-and-rain. The caption search, the direct prompts, the camel
sheet and every call are done. This prompt joins them and judges the hypothesis test.

Objectives
1. Build one table per animal from the caption counts and the calls.
2. Judge the hypothesis test written in the walk map, and write the verdict.
3. Close the walk.

What to read
- artifacts/results/does-a-camel-in-the-rain-come-out-wet/caption-counts.md: from prompts 1 to 3.
- The calls files beside asked-directly.png, a-camel-and-rain.png and the dog sheet: prompt 6.
- The walk map, for the hypothesis test.

What is already known
- The caption counts come from a public stand-in, not SDXL's own data (from prompt 1).
- A verified count is a floor, since captions often leave a wet animal unnamed.
- Each row of the tables is out of 5 seeds, called by me by eye (from prompt 6).

Rules
- Copy the counts and calls as they are. Do not re-call any tile.
- If a count is "not reachable", say so in the table and say what the verdict can still claim.

What to hand back
- The camel folder's README.md, with one entry per sheet saying how it was made, and the tables.
- Each table's columns: verified wet images per word family, "a wet <animal>" out of 5,
  "a <animal> in the rain" out of 5, joint prompt out of 5, PoE out of 5.
- The verdict line: supported, not supported or inconclusive, with the numbers it rests on.
- The walk map updated, and the /drip-retro command the close prints.

What you will see
- At the end, a closing report: what was asked, each output's path, the verdict with its numbers,
  what is still open, and at most one question.

I judge it by eye, from the two sheets, the tables and the verdict line.
```

- `added:` the verified count as a floor (from your point about captions in the wild)
- `dropped:` nothing

## Instructions

*For you to do by hand. Prompts 1 to 3 go in one fresh session, pasted in order.*

Navigation: ⬅️ [Previous](#tasks) | 📋 [TOC](#table-of-contents) | [Next](#quizzes-and-feedback) ➡️

### 1. ✍️ Find a stand-in you can search

You ~15 min (estimate) · Sonnet, medium

Open a fresh Claude Code session in `/home/molef/PhD/poe_repair_min-dog-and-rain` once prompt 4's
job is queued, and paste this.

```text
Background
This repo tests where product-of-experts composition on SDXL fails. Work in the worktree
/home/molef/PhD/poe_repair_min-dog-and-rain on branch exec/dog-and-rain. The hypothesis is that
PoE can only draw an overlap SDXL saw in training, such as a wet camel. SDXL's training data is
text and image pairs, and only the captions are small enough to search. This session finds a
public set of captions I can search by hand.

Objectives
1. Give me step-by-step instructions to find a searchable stand-in for SDXL's training captions.
2. Find out whether it supports a text search, an image search, or both.

What to read
- huggingface.co/stabilityai/stable-diffusion-xl-base-1.0: SDXL's model card.

What is already known
- SDXL's model card names no training dataset, only its text encoders (read 2026-10-20).
- The encoders are OpenCLIP ViT-bigG and CLIP ViT-L (model card).
- The nearest public stand-in is LAION-2B-en, the data OpenCLIP ViT-bigG trained on.
- LAION-5B was withdrawn and re-released as Re-LAION-5B.
- Whether either can be searched from here is unknown, and these instructions find out.
- Counts from the stand-in describe the stand-in, and SDXL's own data stays unknown.

Rules
- Give one instruction per message, then stop and wait for what I saw.
- Each instruction names the site, the click path, what I should see and what it means.
- Write only artifacts/results/does-a-camel-in-the-rain-come-out-wet/caption-counts.md.

What to hand back
- A "Dataset searched" section at the top of caption-counts.md: the dataset, the part of it that
  can be searched, how it is searched, and the date.

What you will see
- Along the way, one instruction at a time, each ending with a question about what I saw.
- At the end, a closing report: what was asked, the file's path, the dataset and its search
  type, what is still uncertain, and at most one question.

I judge it when the section names a dataset I have searched once myself.
```

### 2. ✍️ Count captions for each animal against each word family

You ~30 min (estimate) · same session

```text
Background
This continues the search session. Prompt 1 found the dataset and wrote it at the top of
caption-counts.md. Captions in the wild rarely say "wet", so a wet animal often appears as
"in the rain" or "at the lake". This prompt counts each animal against three word families.

Objectives
1. Give me the query for each animal alone: camel, camels or dromedary; dog, dogs or puppy.
2. Give me the query for each animal with each word family.
3. Saying it is wet: wet, soaked, drenched, soaking, dripping, damp.
4. Weather that implies it: rain, raining, rainy, storm, downpour, puddle.
5. Water contact: swimming, bath, river, lake, splash, beach.
6. Record each count as I report it.

What to read
- artifacts/results/does-a-camel-in-the-rain-come-out-wet/caption-counts.md: the dataset found
  in prompt 1.

What is already known
- The dataset, its searchable part and how it is searched (from prompt 1).
- A rate per animal needs that animal's own count, which is why each animal is counted alone.

Rules
- Give one query per message, then stop and wait for my count.
- Match whole words, ignoring case, so "rain" never matches "train" or "rainbow".
- Use the same dataset and the same part of it for all eight queries.
- Tell me what a hit looks like for each query.

What to hand back
- A counts table in caption-counts.md with eight rows: each animal alone, then with each family.
- Each row holds the query, the count, and the rate (family count over the animal's own count).
- A row that cannot be counted says "not reachable" with the reason.

What you will see
- Along the way, one query at a time, each ending with a request for my count.
- At the end, a closing report: what was asked, the file's path, the eight counts and six rates,
  what is still uncertain, and at most one question.

I judge it when all eight rows hold a count or a reason.
```

### 3. ✍️ Open the hits and mark which really show a wet animal

You ~45 min (estimate) · same session

```text
Background
This continues the search session. Prompt 2 counted captions, but a caption match is not a wet
animal. This prompt opens the images behind the hits, so the count that matters is checked by eye.

Objectives
1. For each animal and each word family, have me open the first 10 hits.
2. Have me mark each image wet, not wet, or can't tell.
3. If prompt 1 found an image search, also run "a wet camel" and "a wet dog" and check the
   top 10 of each the same way.

What to read
- artifacts/results/does-a-camel-in-the-rain-come-out-wet/caption-counts.md: the dataset and the
  eight counts.

What is already known
- The dataset and its search type (from prompt 1).
- The count per animal and family (from prompt 2).
- A verified count is a floor, since captions often leave a wet animal unnamed.

Rules
- Give one family per message, then stop and wait for my marks.
- Record each image's link beside its mark.

What to hand back
- A "Checked by eye" table in caption-counts.md: animal, family, images opened, wet, not wet,
  can't tell, and the links.
- Three example links per animal for the wet images, if there are three.

What you will see
- Along the way, one family at a time, each ending with a request for my marks.
- At the end, a closing report: what was asked, the file's path, the wet count per animal and
  family, what is still uncertain, and at most one question.

I judge it when every family has its images opened and marked.
```

### 6. ✍️ Call every tile wet or dry

You ~45 min (estimate) · the walk session

<a href="../artifacts/results/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain.png"><img src="../artifacts/results/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain.png" width="200" alt="The dog sheet: five seeds down the page, with columns for a dog alone, rain alone, the joint prompt a dog and rain, PoE and SuperDiff"></a> <a href="../artifacts/results/does-a-camel-in-the-rain-come-out-wet/asked-directly.png"><img src="../artifacts/results/does-a-camel-in-the-rain-come-out-wet/asked-directly.png" width="140" alt="The asked-directly sheet: five seeds down the page, with columns for a wet camel, a camel in the rain, a wet dog and a dog in the rain, each prompt rendered alone"></a>

```text
Background
This continues the walk on exec/dog-and-rain. The direct-prompt sheet and the camel sheet have
landed. The dog sheet's walk map records only totals, so its calls per seed are missing.

Objectives
1. Show me each tile one sheet at a time, and record my wet, dry or half call per tile.
2. Call the direct-prompt sheet's 20 tiles.
3. Call the camel sheet's joint, PoE and SuperDiff columns, 15 tiles.
4. Call the dog sheet's joint, PoE and SuperDiff columns per seed, 15 tiles, without changing it.

What to read
- artifacts/results/does-a-camel-in-the-rain-come-out-wet/asked-directly.png
- artifacts/results/does-a-camel-in-the-rain-come-out-wet/a-camel-and-rain.png
- artifacts/results/does-a-dog-in-the-rain-come-out-wet/a-dog-and-rain.png

What is already known
- The dog totals: joint wet on 2 of 5 seeds plus 1 half, PoE on 2 of 5 (walk map, first dog step).
- Seed 10's PoE dog tile became a person with an umbrella (walk map, first dog step).

Rules
- Do not propose a call. I call each tile by eye.
- If my dog calls disagree with the walk map's totals, show me both and ask which stands.

What to hand back
- A calls JSON beside each of the three sheets: seed, column, my call.
- For the dog, the calls file is new and the sheet and tiles stay as they are.

What you will see
- Along the way, one sheet at a time with its tiles numbered, then my calls read back.
- At the end, a closing report: what was asked, the three calls files' paths, the wet count per
  column, what is still open, and at most one question.

I judge it when 50 tiles hold a call.
```

## Quizzes and feedback

Navigation: ⬅️ [Previous](#instructions) | 📋 [TOC](#table-of-contents) | [Next](#later) ➡️

No quiz this phase. The walk's close prints `/drip-retro` for the recall.

## Later

Navigation: ⬅️ [Previous](#quizzes-and-feedback) | 📋 [TOC](#table-of-contents) | [Next](#why-this-chain) ➡️

**Three-concept PoE** (camel, rain and wet together, Composable Diffusion's equation 11 at n = 3):
a different question, whether adding "wet" helps.

**The camel-and-wet and dog-and-wet sheets:** the walk map's fourth dog step still plans the dog's.

**The product-of-experts rule from Bayes,** in the
[sampling-from-a-graph journey](/home/molef/goal-setting/learning/deep-learning/generative-models/03-sampling-from-a-graph/plans/17-poe-rule-from-the-fork.md#1--drip-it-on-the-toy).

## Why this chain

Navigation: ⬅️ [Previous](#later) | 📋 [TOC](#table-of-contents) | [Next](#references) ➡️

<details><summary>Rationale and the time estimates</summary>

**Three kinds of evidence per animal.** The captions say what the stand-in holds. The direct
prompts say what the SDXL in use can draw. PoE is the thing under test. The hypothesis predicts
PoE never draws wet where the direct prompt cannot.

**Two sessions.** The search waits on you at every step and the renders wait on the cluster. The
search session writes only `caption-counts.md`, so the walk map has one writer.

**Times.** All estimates: searching about 4 minutes per query, checking images about 20 seconds
each over 60 images, calling tiles about 30 seconds each over 50 tiles.

</details>

## References

Navigation: ⬅️ [Previous](#why-this-chain) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

The [dog-and-rain walk map](../artifacts/drips/a-dog-and-rain/EXECUTE_WALK.md) and the
[dog sheet's card](../artifacts/results/does-a-dog-in-the-rain-come-out-wet/README.md).

## Next step

Navigation: ⬅️ [Previous](#references) | 📋 [TOC](#table-of-contents)

**Start with:** prompt 4 in a new session, then prompt 5 in the same session, then prompt 1 in a
second session.
