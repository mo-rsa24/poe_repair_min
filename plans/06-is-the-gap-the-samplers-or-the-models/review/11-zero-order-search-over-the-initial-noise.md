# 🎯 Review: zero-order search over the initial noise

This file judges [the design](../plans/baselines/11-zero-order-search-over-the-initial-noise.md):
eight small perturbations of each held-out seed's starting noise, rendered with plain
product-of-experts, the best kept by the validated compose scorer read on the finished image and,
separately, on the model's guess of the finished image at step 10. The random-search row below is
read off existing renders and needs no run. Both runs are done and the verdict is written below;
the bars were written before the first launch.

## Recommended prompt (when the run lands)

```
/analyze-run the noise-search run in prime_lab/poe-repair-animals-compose, group noise-search: the two sheets, compose/*, secondary/* and bothness/*
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/baselines/11-zero-order-search-over-the-initial-noise.md) | the perturbation, the two verifier reads, the sheet, the bars |
| **this file** | **the verdict, and the free random-search row** |
| [plan 10's verdict](10-feynman-kac-steering-on-a-detector-reward.md) | selection among particles mid-run with the same scorer |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [The free row: random search at N 8](#the-free-row-random-search-at-n-8)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Choices made without an answer in the room](#choices-made-without-an-answer-in-the-room)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **The pivot**: the seed's cached starting noise, rendered with plain product-of-experts at DDIM
  `eta` 0. The unperturbed baseline every kept image is judged against.
- **A candidate**: `z' = (z + σ·u) / sqrt(1 + σ²)` with `u` standard normal; still a unit-variance
  Gaussian, at cosine `1 / sqrt(1 + σ²)` to the pivot (0.995 at σ 0.1, 0.958 at σ 0.3). The paper
  never writes this formula; it is this project's reading of "distance λ from the pivot".
- **The kept image**: of the eight candidates, the one with the highest detector count, ties by
  the summed confidence of the detector's kept boxes, then the lowest index.
- **Final-image verifier**: the count read on the finished render. The paper's setting and the
  read the verdict is judged on.
- **Step-10 verifier**: the count read on the decoded x0-hat at step index 10 of 50. The kept
  image under this verifier is still scored on its finished render, so the column asks whether
  picking at step 10 finds a composing finish.
- **Compose rate**: the share of the 8 seeds whose kept image the detector counts two or more
  animal instances in.
- **Both-ness**: the projection of a render's DINOv2 embedding onto the axis from the midpoint of
  the "a cat" and "a dog" clouds toward the "a cat and a dog" cloud, in the landing finding's
  units. PoE sits near 0.21, the joint prompt near 0.52.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#the-free-row-random-search-at-n-8) ➡️

**Establishes a baseline.** A null says eight nudges of this size do not reach a composing noise;
it closes this plan and leaves the iterated climb unrun with that as the reason.

## The free row: random search at N 8

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

Random search is best of N independent noises. The eight held-out seeds 9 to 16 are eight
independent noises, so the existing renders on cat × dog already answer it. Every number below is
read from a file on disk, with the field named.

| Condition | Seeds composing of 8 | Best of 8 composes | Both-ness, mean (min to max) | Nearest cloud is the joint prompt | Source |
|---|---|---|---|---|---|
| plain PoE, λ 0 | **0** (count 1 on every seed) | **no** | 0.21 (0.04 to 0.31) | 5 of 8 | `figure_r32_030050/results.json`, `summary.full.0.0.compose_rate` 0.0 and each `rows[].n_instances`; `cat-x-dog-in-dino-space.json`, `cloud_axes.both_ness_by_condition.poe` and `points[].nearest_cloud` |
| PoE + rank-32 correction, λ 1.0 | 7 (seed 14 fails) | yes | 0.41 (0.31 to 0.50) | not read | same `results.json`, `summary.full.1.0.compose_rate` 0.875; both-ness from the same sidecar, `lora_1.0` |
| PoE + rank-32 correction, λ 1.2 | 7 (seed 14 fails) | yes | 0.42 (0.34 to 0.51) | 8 of 8 | same files, `summary.full.1.2.compose_rate` 0.875 and `lora_1.2` |

**The read.** Random search at N 8 over plain product-of-experts fails on cat × dog: no seed
composes, so no best-of-8 rule can pick a composing one. The best PoE seed by both-ness (seed 16,
0.31) still sits under the worst corrected seed (seed 10, 0.34), so even a verifier that reads
both-ness instead of the count would not find a composing noise among these eight. The corrected
run composes on 7 of 8, so best-of-8 there is trivially a compose. Renders:
`/datasets/mmolefe/poe_repair_min/outputs/showcase/figure_r32_030050/renders/full/seed_<n>_lambda_<λ>.png`;
sidecar: `artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space.json`. Both
read 2026-09-05.

## Runs

Navigation: ⬅️ [The free row: random search at N 8](#the-free-row-random-search-at-n-8) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| Smoke, `GPU=1 bash scripts/noise_search/run_noise_search.sh smoke` over SSH on mscluster107 device 1 (Quadro RTX 8000, `co3`), bash PID 43909, python PID 44055 | Builds a measuring tool | 2026-09-05 18:28 | cat × dog seed 9, N 2, σ 0.3, 10 steps, 512², x0-hat at step 2 | `outputs/interaction_term/noise_search/smoke_N2_s10_20260905-182806/`: `mono_eta0.png`, `poe_pivot.png`, `poe_pivot_xhat_step02.png`, `sigma_0.3/` with `c0.png`, `c1.png`, both `_xhat_step02.png` and `run.json`, the one-row sheet with its sidecar, `summary.json`, `verdict.json`, `history.json`; the log printed `torch_sees=Quadro RTX 8000` and `device cuda`; 51 s for the seed after the model load. Images are noise at 10 steps and 512²; the verdict it wrote is plumbing, not a read. Log `outputs/interaction_term/noise_search/logs/smoke.log` | ✅ done |
| Full, `GPU=1 bash scripts/noise_search/run_noise_search.sh full` over SSH on mscluster107 device 1 (Quadro RTX 8000, `co3`), bash PID 46169, python PID 46316, launch commit `2bae7b0` with a dirty tree | Establishes a baseline | 2026-09-05 18:33 | 2 pairs × 8 seeds × (Mono + pivot + 2 σ × 8 candidates), 50 steps, 1024²; estimated 3.5 h on a Quadro RTX 8000 | run dir `outputs/interaction_term/noise_search/zo_N8_s50_20260905-183320/`, log `outputs/interaction_term/noise_search/logs/full.log`, W&B `prime_lab/poe-repair-animals-compose/runs/3yxrkwgx` (group `noise-search`) | ✅ done 20:27, 1 h 54 min: 16 seed folders, 32 `sigma_*` folders of 16 PNGs plus `run.json`, two sheets with sidecars, `summary.json`, `verdict.json`. The both-ness step failed on the launch node (an xformers attention kernel refusing CPU inputs) and was rerun 20:28 on mscluster85 with `--bothness-only --run-id zo_N8_s50_20260905-183320 --device cpu` and `XFORMERS_DISABLED=1`, which wrote `bothness.json` and updated `summary.json`. Sheets and sidecars filed under `artifacts/results/is-the-gap-the-samplers-or-the-models/noise-search-*` |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] ❓ **Does the kept image, the best of eight perturbed starting noises rendered with plain
      PoE and picked by the compose scorer on the finished image, compose more often than the
      unperturbed seed on cat × dog seeds 9 to 16?** **Inconclusive on the bar, and a null in
      everything but its letter.** `verdict.json`, field `final_verifier`: the pivot composes on 0
      of 8; the kept image composes on 1 of 8 at σ 0.1 (0.125) and 0 of 8 at σ 0.3. The σ 0.1
      difference of 0.125 sits above `NULL_MARGIN` 0.10 and under `PASS_MARGIN` 0.25, which the
      code calls inconclusive. The one seed is seed 16, whose kept image is one body with two
      heads (the eye read below), and the other two counted-2 candidates on that seed are single
      animals. Over all 128 candidates, 3 count two or more, all three on seed 16 at σ 0.1.
      Read from `verdict.json`, field `final_verifier`. Support if the kept rate is at least 0.25
      above the pivot's rate at either σ (`PASS_MARGIN`). Null if within 0.10 at both σ
      (`NULL_MARGIN`). Inconclusive otherwise. The pivot's rate is expected to be 0.0 (the free row
      above), so support means at least 2 of 8 seeds find a composing candidate at one σ, and null
      means at most 0 of 8 at both. Constants in `poe_repair/experiments/noise_search/search.py`:
      `PASS_MARGIN` 0.25, `NULL_MARGIN` 0.10, `SIGMAS` (0.1, 0.3), `N_CANDIDATES` 8,
      `VERIFIER_EARLY_STEP` 10, `ETA` 0. Fixed: one round, plain PoE, 50 DDIM steps, guidance 7.5,
      1024², the cached noise as pivot, the perturbation formula above.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ✅ Did the smoke run produce the Mono reference, the pivot with its early x0-hat, two
      candidates with both reads, a one-row sheet, `summary.json` and `verdict.json`, and did the
      log print the device name and `cuda` from torch? **Yes, all of it** (Runs table, PID 44055).
      The pick rule also did what it says: both candidates counted 1, the final-image pick went to
      the one with the higher summed box confidence (0.652 against 0.584), and every candidate sat
      at cosine 0.958 to the pivot, the formula's value at σ 0.3.
- [x] 🟡 Is the pivot's plain-PoE render within a few grey levels of the existing rank-32 λ 0
      render of the same seed (`mean_abs_diff_vs_r32_lambda0` in each `cell.json`, 0 to 255
      scale)? A large difference means a different starting noise and the run is wrong.
      ❌ on the letter of the plan's stop line, ✅ on what it was checking. Per seed 9 to 16:
      2.2, 9.8, 6.1, **14.5**, 5.4, 6.7, 8.9, 3.9 grey levels (`secondary.pivot_mean_abs_diff_vs_r32_lambda0`).
      The plan set about 10 as the stop line and seed 12 crosses it. Both seed-12 renders were
      opened: the same lying cat-dog in the same pose, colours and background, with a different
      face (a curly-haired blend against a cat's face). The starting noise is the same; the two
      DDIM code paths (this plan's batched step against the LoRA sampler at λ 0) part late in the
      run on this seed. The check as written cannot separate a wrong noise from a late sampler
      divergence, so the layout match is the read, and the line in the plan is corrected to say
      so. Every pivot still counts 1, as every λ 0 render did.
- [x] ❌ Under the step-10 verifier, with the same bars, does the kept image compose more often
      than the pivot (`verdict.json`, field `early_verifier`)? **No: 0 of 8 at both σ, null.** On
      seed 16 the step-10 read counted 1 on all eight candidates, so it could not have picked the
      three the final read counted; its picks there were single animals. The composing candidates
      existed and the verifier was blind to them at step 10. If the final-image verifier supports
      and the step-10 one does not, the composing candidates exist and the verifier is blind where
      the decision is made; if both support, a step-10 read could steer a run instead of only
      picking one.
- [x] 🟡 How often does the step-10 count give the same compose verdict as the final count, per
      candidate (`secondary.sigma_<σ>_early_agrees_with_final`)? **0.922 at σ 0.1 and 1.0 at
      σ 0.3**, and almost all of that is both reads saying "one animal". Among the three
      candidates whose final count was two or more, the step-10 read agreed on none. The number
      is high for the uninformative reason the question predicted. Near 0.5 is guessing. Expected to
      be high only because most candidates fail at both reads; the informative number is the
      agreement among candidates whose final composes.
- [x] ❌ What share of all eight candidates composes, per σ
      (`secondary.sigma_<σ>_candidate_compose_fraction`), and on how many seeds does any candidate
      compose (`sigma_<σ>_any_candidate_composes`)? **σ 0.1: 3 of 64 candidates (0.047), on 1 of 8
      seeds. σ 0.3: 0 of 64, on 0 seeds.** The 0.125 kept rate is one seed with three counted
      candidates, all opened and none two animals. A kept rate of 0.25 from two seeds with one
      composing candidate each says something different from 0.25 with half the candidates
      composing.
- [x] ❌ Does σ 0.3 find composing candidates more often than σ 0.1? **No, the other way: 0 of
      64 at σ 0.3 against 3 of 64 at σ 0.1.** The larger stride (mean cosine to the pivot 0.958
      against 0.995) did not reach a counted candidate on any seed. The prediction was wrong; on
      this evidence the counted candidates are very close to the failing pivot, and they are not
      two animals. The prediction is yes, because
      the larger stride reaches further from a noise that fails; the opposite would say the
      composing noises are very close to the failing ones.
- [x] 🟡 Both-ness of the kept images (`bothness.json`, `by_column.best_final_sigma_<σ>`): do the
      kept images that count 2 sit in the joint band (about 0.5) or the PoE band (about 0.2)?
      **Means over 8 seeds: pivot 0.200, kept by the final verifier 0.235 at σ 0.1 and 0.234 at
      σ 0.3, kept by the step-10 verifier 0.217 and 0.246, against 0.420 for the existing λ 1.2
      renders and 0.523 for the joint prompt** (the same axes reproduce the landing finding's
      0.21 / 0.42 / 0.52 to within 0.01). The kept columns sit in the PoE band. The one counted
      tile, seed 16 at σ 0.1, reads **0.442**, inside the corrected band, while the eye reads one
      body with two heads: the embedding axis reads a two-headed body as "both", which is the
      caveat the landing finding already carries, so this read cannot rescue the count and the
      eye is what settles it. A
      count of 2 with both-ness near 0.2 is a fused animal with two detected parts, and it does not
      count as a compose in the write-up.
- [x] 🟡 Does butterfly × meadow keep its compose rate under the search at both σ? A drop below the
      pivot's is the verifier preferring something other than a butterfly over a meadow. **The
      count is not this pair's read**: a meadow is not an animal, so Mono counts 1 on 5 of 8
      seeds, 2 on one and 0 on two, and the pivot counts 1 on all 8; a count of 2 or more here
      means several butterflies. By eye on the sheet: every kept tile under the final-image
      verifier keeps a butterfly over flowers (8 of 8 at both σ), though the verifier's taste
      shows: seed 14's kept tiles at both σ are a repeating botanical print with three
      butterflies (count 3), and seed 16's at σ 0.3 is one butterfly filling the frame. Under the
      step-10 verifier at σ 0.3, three kept tiles (seeds 10, 13 and 15) finish with no detectable
      animal (count 0); seed 15's is a flat floral pattern with no butterfly. So the step-10
      verifier at the larger stride broke what works on 3 of 8 seeds; the final-image verifier
      did not.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

**Nothing here may ever become a pre-registered threshold**, because anything written here is
written with the answer already visible.

- [x] ❌ Seed 16 at σ 0.1 is the one seed where the search found candidates the detector
      counts two or more animals in: three of eight (candidate 3 count 2, candidate 4 count 3,
      candidate 5 count 2). The kept one, candidate 4, opened while the control pair was still
      running: one body with two heads, a dog-like face in front with its tongue out and a cat's
      head lying behind it on the left. The pivot for the same seed is also a two-headed body.
      By eye this is the fused case the both-ness read exists for, not two animals. The other
      two counted-2 candidates on that seed opened the same way: candidate 3 is one animal with a
      blended face and one body, candidate 5 is one cartoon animal with a collar. So the search's
      only counted successes are one two-headed body and two detector errors, and the both-ness
      number for the kept tile settles whether the write-up counts it (raised by the cat × dog
      compose rates, read 2026-09-05 19:35 before the run finished). Settled after the run: its
      both-ness is 0.442, in the corrected band, so the axis reads the second head as "both" and
      the eye read stands; it is not counted as a compose in the write-up.
- [x] ❌ On that same seed the step-10 verifier read count 1 on all eight candidates, so it
      could not have picked any of the three. The kept candidate's x0-hat at step 10, opened,
      shows one soft animal with one face; its second head is not there yet at step 10 (raised by
      the same read).

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** Every tile in a kept column and the pivot column is a plain
      PoE render from this plan's sampler at 50 steps, guidance 7.5, 1024², `eta` 0; only the
      starting noise differs, at a recorded mean cosine of 0.995 (σ 0.1) and 0.958 (σ 0.3) to the
      pivot. The two reference columns (Mono from the same sampler, the existing λ 1.2 render
      from the LoRA sampler) are named as references on the sheet and never entered a verdict.
- [x] 🟡 **Was the measuring tool sound?** The detector scored all 128 candidates at both reads
      and the sheet draws its count on every tile. The verifier and the judge are the same scorer
      and it was selected for exactly as feared: of the 3 counted candidates on cat × dog, two are
      single animals the detector boxed twice and one is a two-headed body; on the control pair
      it favoured multi-butterfly prints. The both-ness read did not catch the two-headed body
      (0.442, in the corrected band). The eye read caught all of it, and it is recorded above.
- [x] ✅ **Did the run respect the environment?** Everything under
      `/datasets/mmolefe/poe_repair_min/outputs/interaction_term/noise_search/`, `co3` on
      mscluster107 (a Quadro RTX 8000), `torch_sees=Quadro RTX 8000` and `device cuda` in the
      log, device 1 at 20 MiB and 0% before both launches with no process of ours on it, node,
      device and PIDs in the Runs table; 7 min per seed for 18 renders, so nothing fell back to
      the CPU. The only fallback was the both-ness step, which failed loudly and was rerun.

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#choices-made-without-an-answer-in-the-room) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| a search over the starting noise composes, or does not | the kept rate per σ against the pivot's, with N, the step count, `eta` 0 and the seed count |
| the method follows Ma et al. | that the paper never writes the perturbation, that this one is variance-preserving, and that one round at N 8 is the smallest version of their zero-order search rather than their iterated climb |
| the verifier was the compose scorer itself | that the kept image is chosen by the scorer that scores it, and the both-ness read that separates two bodies from a fused face |
| random search at N 8 fails on plain PoE | that the eight noises are the held-out seeds already rendered, not fresh draws |
| the search found one counted success in 128 candidates | that it is one body with two heads by eye, that both-ness reads it as "both" (0.442) and so cannot be the arbiter here, and that the other two counted candidates are single animals |
| the pivot is the plain PoE render every other figure uses | that this plan's batched DDIM and the LoRA sampler at λ 0 part late on some seeds (up to 14.5 grey levels on seed 12, same scene, different face), so the pivot column is this sampler's plain PoE and not byte-identical to the λ 0 column elsewhere |
| the composing control pair was not harmed | that the count is not that pair's read (a meadow is not an animal), that the final-image verifier kept a butterfly on 8 of 8 at both σ, and that the step-10 verifier at σ 0.3 lost it on 3 of 8 |

## Choices made without an answer in the room

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

This plan was designed and integrated in an unattended session on 2026-09-05. Where a skill would
have asked, the recommendation was taken and is recorded here.

**The plan was written in plan 10's shape and queued for the tree rather than walked through
`/integrate-plans`.** The shared-files rule forbids editing the scope's plans table and the root
running order, so the integration is the plan file, this review file, and two rows in the
pending-sync list; one sync folds them in.

**The perturbation formula is the variance-preserving mixture**, because the paper gives none and
this is the one form under which a candidate is still a legitimate starting noise for the sampler.

**One round at N 8, not the paper's iterated climb**, because one round decides whether the
neighbourhood holds composing noises at all, and it is what the session's prompt asked for.

**The step-10 verifier is a second selection rule on the same candidates, not a second render.**
The sampler is deterministic, so rendering twice would produce the same eight images; capturing
x0-hat at step 10 during the one render is the same experiment at half the cost.

**Tie-break by summed box confidence, then index.** Counts are integers and most candidates tie at
1; without a second key the pick would be the first candidate by index, which is a random noise.

## Still open

Navigation: ⬅️ [Choices made without an answer in the room](#choices-made-without-an-answer-in-the-room) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

- [ ] The paper's iterated zero-order search (best candidate becomes the next pivot) is not run:
      one round found no two-animal candidate to climb from, and a climb toward what the counter
      rewards here (two-headed bodies, multi-butterfly prints) is a climb toward detector errors.
      It would only be worth running with a verifier that cannot be satisfied by a second head.
- [ ] The both-ness read is in `bothness.json` and the repo copy, and not in the W&B run
      (`3yxrkwgx`), because the rerun that produced it does not log; the sheets and the other
      sidecars are in the run's artifact.
- [ ] The identity check's stop line in the plan is corrected to name what it can and cannot tell
      apart; the numbers themselves stay as measured.

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

The finding is written at
[does searching over the initial noise compose](../../../report/is-the-gap-the-samplers-or-the-models/does-searching-over-the-initial-noise-compose.md).
It is the model-side reading: eight nudges at cosine 0.96 to 0.995 of a failing noise do not
reach a two-animal image, and the one counted success is a second head. It joins plans 09 and 10
as the third selection-only baseline that does not compose, and it is read against the in-span
share once scope 05's rung 2 lands.
