# 🧪 Review: does choosing the injected noise in the commit window give the corrected sampler a crisper image with the two animals kept?

**Both runs are done and judged: null on the pre-registered bar, control pair intact.** This file judges [the design](../plans/experiments/16-search-the-noise-in-the-commit-window.md). Its answer fills the showcase wall's "noise searched" row if supported, and otherwise turns the paper's fidelity caveat into a statement about what inference compute cannot buy.

## Recommended prompt (when the run lands)

```
/analyze-run <run id>
```

## Position in the plan tree

| File | What it holds |
|---|---|
| [design](../plans/experiments/16-search-the-noise-in-the-commit-window.md) | the sampler, the search, the reward, the code, the constants |
| **this file** | **the verdict: whether the searched run is crisper than the unsearched one with the animals kept** |

## Table of contents

- [Words this file uses](#words-this-file-uses)
- [Run kind](#run-kind)
- [Runs](#runs)
- [The question written before the run](#the-question-written-before-the-run)
- [Written before the run, answered after](#written-before-the-run-answered-after)
- [Asked after the result](#asked-after-the-result)
- [Could the answer be an artefact](#could-the-answer-be-an-artefact)
- [What the write-up owes](#what-the-write-up-owes)
- [Still open](#still-open)
- [Next step](#next-step)

## Words this file uses

Navigation: 📋 [TOC](#table-of-contents) | [Next](#run-kind) ➡️

- **The control**: the corrected sampler (PoE plus 1.2 × the rank-32 step-30050 correction on every step) run stochastically at eta 1, with the noise injected at each step taken from one seeded stream.
- **The searched run**: the same sampler on the same stream, except that at steps 8 to 25 each draw is the starting point of a greedy search over candidate draws (three rounds of three, a fresh draw with probability 0.25, otherwise the current best nudged at σ 0.3), each candidate judged on the decoded running estimate one step ahead.
- **The reward**: the validated instance count clipped at 2, plus half a sigmoid of ImageReward (a learned human-preference score against "a cat and a dog"). The count decides the level; the preference score orders within a level.
- **Sharpness**: Laplacian variance of the greyscale 1024 px render, higher is crisper, the function plans 07 and 14 used. Read **paired per seed** (searched against control on the same seed), never as a band, because the sketch seeds set any band's edges.
- **Compose count**: seeds of 8 where the detector counts two or more animal instances.
- **Both-ness** and **distance to Mono**: the render's DINOv2 embedding projected toward the joint-prompt centroid on the landing finding's axes, and one minus its cosine to the seed's joint-prompt render.
- **Butterfly present**: on the control pair, a "butterfly" box at confidence 0.30 or more.

## Run kind

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

**Tries a new idea to improve the result** (group 2: a sampler-side method from a named paper, arXiv 2506.03164, on the shipped adapter, no training). A missed bar closes the plan and the fidelity caveat is written with the sampling cost this run measured; a support lands as one wall row and never rewrites the hypothesis.

## Runs

Navigation: ⬅️ [Run kind](#run-kind) | 📋 [TOC](#table-of-contents) | [Next](#the-question-written-before-the-run) ➡️

| Run | Kind | Launched at | Cost | Output | State |
|---|---|---|---|---|---|
| smoke: cat × dog seed 9, 10 steps, 512², search at steps 2 to 5, one round of 2 | builds a measuring tool | 2026-09-06 03:00, Slurm job 50333 on `bigbatch` mscluster54's neighbour mscluster53 (RTX 3090, `co3`; every `biggpu` device was carrying a process), `sbatch --nodelist=mscluster53 scripts/noise_trajectory_search/noise_trajectory_search.sbatch smoke` | 82 s | `/datasets/mmolefe/poe_repair_min/outputs/showcase/noise_trajectory_search/smoke_nts_lam1.2_s10_20260906-030024/`, log `logs/nts-50333.out` | ✅ done: five renders, `search_eta1/run.json` with 12 evaluations over 4 searched steps (pivot won 1, local 2, fresh 1), sheet, strip and the four figures drawn |
| full: both pairs, seeds 9 to 16, 50 steps, 1024², search at steps 8 to 25, three rounds of three | tries a new idea | 2026-09-06 03:07, Slurm job 50337 on `bigbatch` mscluster54 (RTX 3090, `co3`), `sbatch --nodelist=mscluster54 scripts/noise_trajectory_search/noise_trajectory_search.sbatch full` | 2 h 37 min on the RTX 3090, about 9 min per seed (180 candidate evaluations each) | `/datasets/mmolefe/poe_repair_min/outputs/showcase/noise_trajectory_search/nts_lam1.2_s50_20260906-030442/`, log `logs/nts-50337.out`, W&B run `mqtwuoem` (group `noise-trajectory-search`); the sheets, strips, figures and summaries filed under `artifacts/results/does-searching-the-noise-in-the-commit-window-sharpen-the-fix/` | ✅ done |

## The question written before the run

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#written-before-the-run-answered-after) ➡️

- [x] ❌ **On cat × dog, is the searched run's finished render sharper than the control's on at least 6 of 8 seeds, with the compose count within one seed of the control's?**
      The bar, in source as `SUPPORT_MIN_SHARPER_SEEDS = 6`, `SUPPORT_MAX_COMPOSE_LOSS = 1`, `NULL_MAX_SHARPER_SEEDS = 4` and `NULL_MIN_COMPOSE_LOSS = 2` in `poe_repair/experiments/noise_trajectory_search/search.py`, applied by `search.verdict`:
      **support** if `compose_n(search) ≥ compose_n(control) − 1` and the searched render is sharper on at least 6 of 8 seeds;
      **null** if the compose count falls by 2 or more (the animals go with the search) or the searched render is sharper on 4 or fewer seeds (a coin flip or worse);
      **inconclusive** at exactly 5 of 8 with the animals kept.
      Under a null of no effect, 6 or more of 8 happens by chance about 14% of the time; eight seeds cannot do better, and the paired figure shows every seed so the reader can weigh it.
      Answer: **null.** Compose count 8 of 8 for both the control and the searched run (the animals are kept), but the searched render is sharper than the control's on **3 of 8** seeds (9, 11, 13; per seed control against search: 61/78, 38/35, 126/181, 170/87, 27/28, 114/76, 302/268, 22/12). Mean sharpness 96 against the control's 107. By eye the searched column is at least as crisp on most rows and the number disagrees on seeds 12, 14, 15 and 16, which is the edge-counting measure and not a reason to move the bar. From `summary_a_cat__x__a_dog.json`, fields `paired_search_vs_control.sharper_seeds` and `conditions.*.per_seed`.

## Written before the run, answered after

Navigation: ⬅️ [The question written before the run](#the-question-written-before-the-run) | 📋 [TOC](#table-of-contents) | [Next](#asked-after-the-result) ➡️

- [x] ✅ **Does the search find anything at all, or does the control's own draw win most steps?** From `summary_<pair>.json`, field `secondary.chosen_kind_counts`. If the pivot wins on most of the 144 searched steps (18 per seed × 8), the reward cannot see a difference between draws at these steps and the verdict, whatever it is, is about the reward's blindness rather than the noise. Reported either way.
      Answer: it finds something. The pivot won **6 of 144** searched steps on cat × dog (local nudge 111, fresh draw 27; on the control pair 7 / 108 / 29), with a mean reward gain of 0.09 per step (0.32 on the control pair). The count term was 2 on nearly every candidate, so the gain is ImageReward's sigmoid, about a quarter of its range.
- [x] ❌ **Does a gain in the reward one step ahead show up as a sharper final?** From `paired_search_vs_control.sharper_seeds` against `secondary.mean_reward_gain_per_searched_step`. A large mean gain with few sharper seeds says the reward's preference term prefers something other than crispness; a small gain with many sharper seeds says the noise matters more than the reward sees.
      Answer: no. The reward rose at every searched step and the final is sharper on 3 of 8; the reward's own term is higher on the final on 5 of 8, both-ness on 5 of 8, and the searched final is nearer the joint-prompt render on 6 of 8. The search buys the semantic reads it was pointed at and not the edge count.
- [x] ✅ **What does eta 1 alone do to the shipped render?** The control against the eta-0 render on the same seeds: compose count, mean sharpness, both-ness. This is the axis the search sits on top of, and a control that already loses animals at eta 1 changes what a support means.
      Answer: eta 1 alone is the larger effect. The eta-1 control composes **8 of 8** against the shipped eta-0 render's 7 (seed 11 gained), and is sharper on 6 of 8 seeds, mean 107 against 57; both-ness 0.39 against 0.44 (higher on 3 of 8). This changes the reading: the crisp two-animal column on the sheet is mostly the stochastic sampler, and the search adds the semantic margin on top. From `conditions.control_eta1` and `conditions.adapter_eta0`.
- [x] ✅ **Does the searched run stay in the corrected basin?** From the tracks figure and `paired_search_vs_control.higher_both_ness_seeds`: the searched track should end inside the corrected band of both-ness (about 0.42 in the landing finding, against 0.20 for plain PoE), not fall back toward the plain-PoE band.
      Answer: yes. Final both-ness mean 0.43 for the searched run, 0.39 for the control, 0.44 for the shipped render, 0.23 for plain PoE; higher than the control on 5 of 8 seeds. In the tracks figure the searched tracks climb into the joint-prompt cloud inside the window and end there; the over-steps figure puts the searched run's both-ness at about 0.35 by step 10 where the control reads 0.25.
- [x] ✅ **Does the search break the control pair?** Butterfly present on at least 7 of 8 seeds in the searched column, by `search.control_verdict` (`CONTROL_MAX_PRESENCE_LOSS = 1`).
      Answer: intact. Butterfly present on **8 of 8** in every column; sharper than the control on 6 of 8 on this pair. The correction turns the meadow into an illustration in all three corrected columns (mean sharpness 464 to 539 against 65 for the references), which the adapter does on this pair with or without the search.

## Asked after the result

Navigation: ⬅️ [Written before the run](#written-before-the-run-answered-after) | 📋 [TOC](#table-of-contents) | [Next](#could-the-answer-be-an-artefact) ➡️

Questions the result itself raised. **Nothing here may ever become the question above**, because it was
written with the answer already visible.

- [ ] 🟡 **Is Laplacian variance the wrong instrument for this question?** (raised by the sheet: seeds 12 and 16 read the searched tile as softer by a factor of two while both tiles are in focus, and plan 14 hit the same disagreement on its sketch seeds.) The bar stands as written. A paired eye read, or a no-reference sharpness measure that ignores line density, would make the next such run judgeable; nothing here is re-judged on it.
- [ ] ⚠️ **Is the stochastic sampler at eta 1 a better shipped setting than eta 0?** (raised by the control column: 8 of 8 composing and sharper on 6 of 8 than the eta-0 render on the same seeds.) One pair, one eta, no seed repeat; a cheap follow-on is eta 0.5 and 1.0 on the 8 seeds through the same sampler with no search.

## Could the answer be an artefact

Navigation: ⬅️ [Asked after the result](#asked-after-the-result) | 📋 [TOC](#table-of-contents) | [Next](#what-the-write-up-owes) ➡️

- [x] ✅ **Was the comparison fair?** Yes. The control and the searched run share the sampler, the checkpoint, λ, eta, the initial noise and the noise stream; only the draws chosen inside steps 8 to 25 differ. The counts confirm 8 seeds per condition, and `evaluations` in each `search_eta1/run.json` confirms the search ran at every window step: 180 evaluations on every seed of both pairs, 8 cells per pair.
- [x] 🟡 **Was the measuring tool sound?** Partly. The judge (paired sharpness, the compose count, both-ness) is not the reward (count plus ImageReward). Sharpness counts edges; the paired read exists because of that, and the eye read on the sheet is recorded beside it: the sharpness number and the eye disagree on four of eight searched tiles, which is the first question under Asked after the result.
- [x] ✅ **Did the run respect the environment?** Yes. Output under `/datasets`, the launcher's guards passed inside the job, node, job id and PID in the log header and here, `torch.cuda.is_available()` true on the pinned device (mscluster54 device 0, PID 1047237, disk at 34%).

## What the write-up owes

Navigation: ⬅️ [Could the answer be an artefact](#could-the-answer-be-an-artefact) | 📋 [TOC](#table-of-contents) | [Next](#still-open) ➡️

| What the paper says | What it owes alongside it |
|---|---|
| a "noise searched" row on the wall, if supported | the window in steps, the reward, the number of extra forward passes per image, and that the read is paired sharpness on one pair |
| the fidelity caveat, if null | that a per-step noise search of this size did not buy crispness, with the cost stated, so the caveat is bounded rather than vague |

## Still open

Navigation: ⬅️ [What the write-up owes](#what-the-write-up-owes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| What is unresolved | What would settle it | Who or what is blocked by it |
|---|---|---|
| whether a larger search (the paper's 4 candidates × 20 rounds at every step) changes the answer | one seed at the paper's budget, about 4,000 corrected forwards | nothing; this plan's budget is the one a showcase render could afford |

## Next step

Navigation: ⬅️ [Still open](#still-open) | 📋 [TOC](#table-of-contents)

Instruction 5 in the design: the eye read on the sheet, then the close-out. The null goes to the paper's fidelity caveat with the cost stated: 180 extra corrected forward passes per image bought the semantic reads and not the edge count, and eta 1 alone bought more.
