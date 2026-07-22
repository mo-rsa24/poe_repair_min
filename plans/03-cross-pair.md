# ↔️ Cross-Pair — a pair-trained fix transfers to an unseen sibling pair

## Description
Take the LoRA already trained on cat×dog (the G6 pool, run `pueuo7bl`) and, with no retraining, run
it on cousin pairs from the same group (`a_wolf__x__a_husky`, `a_lion__x__a_dog`, …) on seeds it
never saw. Folds phase file 12.

## Purpose
Serves Objective 3 (Cross-Pair) and Definition-of-Done item 3. It is the cheapest check of whether
the difficulty grouping actually predicts transfer. It also feeds the Group-Wise and Scale rungs: a
cousin pair that fails here is a candidate to add to the Scale training pool.

## Goal
For each group, a pass/fail on transfer to a cousin pair: does the group's LoRA show both concepts
on at least 2 of 4 unseen seeds?

## Tasks
- [x] ✅ Land the `--heldout-pair` sampler flag + `build_eval_cache.py` + `render_seed_summary` override. → code ready (Plan 12).  ✓ verified (sample_heldout.py --heldout-pair, scripts/build_eval_cache.py)
- [ ] ⚠️ Build the sibling-pair eval caches (seeds 9–12), then run the held-out-pair driver.
  Prompt: `POE_REPAIR_TRAINING_CACHE=/datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache bash scripts/cross_seed_lora_pooling/build_sibling_caches.sh && bash scripts/cross_seed_lora_pooling/heldout_pair.sh` (note: siblings like `training_cache/heldout/a_wolf__x__a_husky` already exist as minimal eval stubs — verify before rebuilding).
- [ ] ⚠️ Direct G6 run:
  `$PY -m poe_repair.experiments.cross_seed_lora_pooling.sample_heldout --checkpoint artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/taskB__k04_ep2000_resumed__wandb-pueuo7bl/checkpoints/lora_step_100000.pt --pair a_cat__x__a_dog --heldout-pair a_wolf__x__a_husky --out-dir artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/heldout_pair/a_wolf__x__a_husky`
- [ ] ⚠️ Render `render_heldout_summary.py`; classify transfer per sibling; flag pairs that fail as Scale-pool candidates.
- [ ] ⚠️ Read this rung as a SMOKE TEST, not transfer evidence: single-pair → sibling is confounded — a cat×dog-only LoRA saw no variety, so a hit can't be told from "the memorised correction happens to fit." The reviewer-credible transfer test is Plan 16 ([04-group-wise.md](04-group-wise.md)) with concept-disjoint siblings.
- [ ] ⚠️ Delivery boost, inference-only (graft B1): if a cousin pair barely composes, add Attend-and-Excite on top of the LoRA at sampling time — it nudges the model to keep every subject visible, no retraining. Always score LoRA-only vs LoRA+A&E, so the LoRA gets the credit, not the nudge.
  How: add an `--attend-excite` flag to `sample_heldout` (a small latent step on whichever of "a cat"/"a dog" is being ignored each step; code from github.com/yuval-alaluf/Attend-and-Excite), then rerun the direct G6 cousin sample with and without it.

## Recommended skill
— custom; no skill fits (sample_heldout `--heldout-pair` + `render_heldout_summary`; use `/run-experiment` for the cache build).

## Engagement Instructions
```
$ ls artifacts/rung2-survive-noise/cross_seed/a_cat__x__a_dog/heldout_pair/a_wolf__x__a_husky/sample_seed_*.png
# Expect samples on seeds 9–12; summary <train>__heldout__<eval>.png shows the sibling composing (or not) beside PoE/Mono refs.
```
