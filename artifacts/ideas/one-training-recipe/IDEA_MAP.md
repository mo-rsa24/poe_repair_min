# 💡 One training recipe for a clean, composing adapter

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1. The good pictures come from the weights, not from the draw | needs a check | run 1 (job 59686): raw twice and EMA on four checkpoints; the rank-8 repeat already came back bit-identical |
| 2. Composition is already had; only fidelity is missing | wrong as stated | a species read of the v58-05 sheets: two animals in every cell, the right two species on most cat×dog seeds and on few tiger×dog or cat×fox cells; misses sit on shared-body-plan pairs, the adapter beats its teacher on chess×hourglass |
| 3. A per-checkpoint judge that sees species can be fixed before the run | needs a check | **a per-instance species read (validated instance count, then CLIP over the two concept names per box) checked against this session's 30 by-eye labels** |
| 4. The failures have a cause a training change can reach | needs a check | **haze: the adapter copies the damping half of the correction (each expert 7.5 to 1-3) and only half the rest, so the out-of-span-only run (v58f-07, died at step 541) is the test. Species: the four unread empty-branch runs and the unread attention-intervention result** |
| 5. An added training term buys fidelity without costing composition | needs a check | **two terms, two runs: the attention loss adapted to two single-token branches (masks from the targets first), and a reward run with the four fixes refl-03 lacked (worse-of-two identity, ImageReward on, an anchor to the distillation adapter, EMA, held-out pairs)** |
| 6. Changing one axis at a time from v58-05 converges on one recipe | wrong as stated | **sequential single changes do not fit the time, and the walk found two levers (species, picture) not one; repaired as a batch on the two free cards: two single-change runs now, the combined run when masks exist, the reward last, a second training seed of the winner before "reliable" is claimed** |

Load-bearing: claim 1. "Reliably after N steps" means nothing if the good pictures are a lucky
seed at a lucky checkpoint.

## Table of contents

- [Position in the idea](#position-in-the-idea)
- [Quick context: where you are](#quick-context-where-you-are)
- [The idea, as it stands](#the-idea-as-it-stands)
- [The claims](#the-claims)
- [What the words are](#what-the-words-are)
- [Held claims](#held-claims)
- [Dead ends](#dead-ends)
- [Checks outstanding](#checks-outstanding)
- [Runs](#runs)
- [Routes](#routes)
- [Sources](#sources)
- [Next step](#next-step)

## Quick context: where you are

Navigation: ⬅️ [Position](#position-in-the-idea) | 📋 [TOC](#table-of-contents) | [Next](#the-idea-as-it-stands) ➡️

**What the idea is**

One training recipe for the correction adapter, run for a fixed number of steps with a stopping
rule written in code, that gives clean two-object PoE renders on held-out pairs and seeds.

**Where the walk is**

Compiled 2026-09-25 (destination A). Training runs R1, R2a and R2b started that day through
`scripts/launch_recipe_run.sh`; R0 waits for mscluster110, which answers only an interactive
SSH; R4 waits for two additions to the reward trainer. Run 1's EMA half was still rendering. The judge's second pass reached 13 of 19 and can rank but not stop. Run 1 still rendering. Claim 1 waits on run 1 (job 59686, running on mscluster125).

**What compile produced**

Destination A. The load-bearing claim's repeat half is settled (the render is deterministic on
one card) and its EMA half is a named, running check. The recipe is a stack of two levers
with a judge that ranks but does not yet stop, and the batch that decides it is running.

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

Train the correction adapter by distillation (regressing the adapted PoE prediction onto the frozen
model's joint-prompt prediction), starting from the v58-05 recipe, and change one thing at a time until a fixed-length run reliably yields an adapter whose PoE renders on
held-out pairs show two separate, correctly-speciated objects at the base model's photo quality.
"Reliably" covers three sources of variation: the sampling seed, the checkpoint, and the training
run itself. Candidate changes are EMA (Polyak) weights, freezing or anchoring the empty branch,
training on the adapter's own states (on-policy distillation), a KL-anchored picture reward, and a
token-level cross-attention loss against subject mixing. Which sampling steps the loss covers is
itself a free choice, not a fixed part of the recipe.

Training runs go on `biggpu` (RTX 8000 and A6000 on mscluster106, 107, 108, 109 under `co3`)
or the Blackwell cards (mscluster110 and 112 under `co3_bw`), with bigbatch 3090s allowed for
the overflow. Every `biggpu` node is allocated in Slurm and the queue is about twelve deep, so
runs start through the shared-device path (`nohup` on a free card, per
`environment/hpc/execution-protocol.md`). Probed 2026-09-25: 106 GPU 1, 107 GPU 1 and 109 GPU 1
empty; 112 carries a foreign process; 111 reads "GPU requires reset" and torch sees no device
under `co3_bw`; 110 answers SSH from the session node with no output, unverified. A species-aware judge, fixed before the
run, picks the stopping step.

## The claims

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

### 1. 🔍 The good pictures come from the weights, not from the draw

Mark: needs a check. Review 06 in plan 09 already shows elephant×penguin seed 9 as an engraving at
every checkpoint of `pool43-all50` and seed 10 photographic at every one, so the seed carries the
style. Whether the checkpoint swings are also the render rather than the training is unmeasured:
the EMA weights have never been rendered (`scripts/sweep_probe.sbatch` never passes `--ema`), and
no recipe has been trained twice with a different training seed.

### 2. ❌ Composition is already had; only fidelity is missing

Mark: wrong as stated. Read by eye on
`artifacts/results/which-checkpoint-to-take-each-figure-cell-from/latest/which-checkpoint-v58-05-self-contrast_seeds_9_42.png`,
the three distinct-animal pairs (cat×dog, tiger×dog, cat×fox) at seeds 9 and 42 over the four
checkpoints 10k, 20k, 30k, 37.5k (24 cells): two animals appear in every cell, and both are the
right species in about 6 (cat×dog seed 9 at all four checkpoints, tiger×dog seed 42 at 10k, and one
or two more). Cat×dog seed 42 draws two dogs at every checkpoint; tiger×dog seed 9 draws a
dog-tiger hybrid beside a tiger; cat×fox draws a fox beside a second fox or an unreadable dark
animal. The instance-count compose rate reads 1.0 on all of these.

Repaired wording: two objects reliably appear. Whether each is its own species is unsolved and
fails on most held-out cells, and the hybrid and wrong-species failures are composition failures
under that reading. What is left for "fidelity" is picture quality: haze, resolution, drawn
style, and swings between checkpoints. The cost of the repair: the judge in claim 3 has to score
species per object, and every "composes 8 of 8" result in the tried list is a count of animals,
not of concepts.

Read wider on cat×dog alone (seeds 3, 5, 9, 13, 15, 42 across the four `seeds_*` sheets): the
adapter draws a cat and a dog on 5 of 6 seeds, drawn rather than photographic on seeds 5 and 15.
The species failures concentrate on tiger×dog and cat×fox, and on cat×fox the joint-prompt
teacher itself draws foxes and no cat on 4 of 6 seeds
(`report/is-the-adapter-just-copying-the-joint-prompt/does-the-joint-prompt-always-draw-both-concepts.md`).

- [x] 2.1 why cat×dog seed 42 never draws the cat when seed 9 always does. Settled: it is the
  adapter, not the seed. v58-03 (same recipe without the 0.01 contrast term) draws a white cat
  beside a golden dog on seed 42 at 30k, 40k and 50k (`ablation_grid/a_cat__x__a_dog/seed42/ours_v58-03-self_s0*`),
  while v58-05 draws two dogs at every checkpoint. The plain-PoE hybrid's look does not predict
  it either: seed 13's hybrid is a white dog and the adapter still draws an orange cat there.
- [x] 2.2 whether the teacher's own failures on a pair are what the adapter reproduces. Wrong as
  stated. Training targets are excluded by construction: every v58 target was inspected by hand
  for clean co-occurrence. On the evaluation side the teacher fails on both cat×fox (foxes, no cat)
  and chess×hourglass (board, no hourglass), and the adapter fails on the first and succeeds on
  the second: a clear hourglass beside chess pieces on both seeds at every checkpoint of v58-05
  (`which-checkpoint-v58-05-self-contrast_seeds_9_42.png`, columns 11 and 12). So the adapter is
  not bounded by its teacher on unseen pairs. Repaired: the species misses concentrate on pairs
  whose two concepts share a body plan (cat×fox, tiger×dog, cat×dog on one seed), and shape-distinct
  pairs compose even where the teacher cannot. That is the subject-mixing pattern the attention-loss
  literature describes, and it is claim 4's first lead.

### 3. 🔍 A per-checkpoint judge that sees species can be fixed before the run

Mark: needs a check. The pieces exist: the validated instance count
(`poe_repair/experiments/compose_scorer_validation/detection_scorer.py`, GroundingDINO query
"animal", NMS, `MIN_BOX_FRACTION` 0.25) says how many bodies; CLIP already loaded in
`poe_repair/rewards/plurality.py` can label each body's crop against the two concept names. The
composed-by-species read is: at least two bodies, one labelled each concept. Its check is
agreement with the by-eye labels from claim 2 (24 cells on the seeds 9 and 42 sheet, 6 more on
the cat×dog strip). The rejected both-ness axis and the joint-prompt-distance read stay
rejected. The plurality reward's identity term averages the two concepts' CLIP scores
(`plurality.py:195`), so a strong elephant pays for a missing penguin; a judge takes the weaker
of the two.

### 4. 🔍 The failures have a cause a training change can reach

Mark: needs a check, split by the two failure kinds from claim 2.

Picture quality (haze, low contrast). The cause is half known. The true correction over steps
0 to 10 is 63% re-weighting (each expert's guidance turned down from 7.5 to 1 to 3) and 37% a
direction no re-weighting reaches (`report/when-does-the-outcome-lock-in/what-is-the-correction-made-of.md`,
0.374, inconclusive bar). The rank-32 adapter reproduces the re-weighting part at cosine 0.9 and
only half the other direction. Turning guidance down is what low contrast looks like, and a plain
3/3 re-weight at render time smears both subjects on 8 of 8 cells, so the re-weighting part
cannot be dropped at render time either. The training change that reaches this exists as a flag,
`--out-of-span-only 1` in `train_pooled.py`, with the damping applied at sampling by
`--expert-weights`; its one run (v58f-07) died at step 541 and has never been read.

Species misses (two of one, hybrids). The lead from claim 2 is subject mixing on shared-body-plan
pairs. Two unread results bear on it: the empty-branch runs (freeze-null `kdzx03ji`,
adapt-null-only `qbvj0wsm`, null anchor `72as6yk3`, v58-04-freezenull `mwrgppyw`), and the
attention intervention `poe_repair/experiments/mechanism_study/aae_intervene.py`. The empty
branch carries where, not what (it enters at −14 and the loss cannot see it). Run 3 read the
frozen-branch run: species no better, picture worse (drawn, crowded, invented people), and
adapting only that branch was already recorded as one-animal clip art. Neither branch change
reaches the species misses, so that fix is token-level. The attention capture under
`outputs/attn_mechanism/aae_equiv/a_cat__x__a_dog/` (seeds 9 and 10, step size 20) is still
unread and is a capture of maps, not a sheet.

Drawn style. Carried by the seed on identical weights (elephant×penguin seed 9 an engraving at
every checkpoint, seed 10 a photo at every one), and made worse by broad pools. No training
change on a fixed pool reaches it, so the recipe does not promise it and the judge compares per
seed.

### 5. 🔍 An added training term buys fidelity without costing composition

Mark: needs a check, one per term. The evidence that a term can cost composition is already
in hand: the 0.01 contrast term is the one axis between v58-03 and v58-05, and on cat×dog seed
42 it costs the cat at every checkpoint (claim 2.1). So each term is judged on both counts.

The attention loss. The published form (TokenCompose) ties each subject token's cross-attention
to that subject's mask inside one prompt. Here each branch carries one token, so the loss
becomes: in the cat branch, the "cat" token's attn2 map over the latent stays inside the cat's
region of the target; in the dog branch, likewise. It asks the adapter to make each branch
spatially selective, which is the "where" the empty branch was found to carry. Needs masks
per concept per target (GroundingDINO boxes, or Grounded-SAM), which nothing builds yet; the
attention capture code (`mechanism_study/capture_attention.py`, attn2 q,k,v hooks) is the
place the maps already come out of.

The reward. `reward_finetune/train.py` has no anchor, no EMA and the mean identity term, which
is the refl-03 setup that rose through distinctness alone. The fair run adds: identity as the
worse of the two concepts, ImageReward on, an L2 or KL anchor to the distillation adapter it
starts from, EMA, and pairs held out of the reward. Its cost is a render inside every step,
which is why refl-03 managed 1,331 steps before dying.

The repair for the attention loss (from `break`): weight the existing distillation error by
region, with the weaker object's mask (the smaller box, or the one the judge scores lower)
weighted up. Same trainer, same target, only the per-pixel weight changes, and it pushes on
the content inside the region where the hybrid lives. The masks it needs come from the instance detector's "animal" boxes on each cell's `mono.png`
in the training cache, each body labelled by the judge's CLIP read, because run 5 showed the
per-concept queries land on the same body on hybrids.

### 6. ❌ Changing one axis at a time from v58-05 converges on one recipe

Mark: wrong as stated, for two reasons. Run one change, wait, run the next does not fit the
time, and the recipe is a stack of two levers (one for species, one for picture quality), so
the end state was never one axis away. Repaired as a batch:

| Slot | Run | The one change against v58-05 | Aimed at | Starts |
|---|---|---|---|---|
| now, mscluster109 GPU 1 (A6000) | R1 | `--out-of-span-only 1`, damping applied at render by `--expert-weights` (the v58f-07 rerun, after reading why it died at step 541) | haze, claim 4 | today |
| now, mscluster106 GPU 1 (RTX 8000) | R2 | the on-policy `--adapter` cache as the training states | haze and hybrids, claim 4 | today |
| next free card, after the mask job | R3 | the winner of R1/R2 plus the region-weighted distillation loss, masks from each cell's target | hybrids, claim 5 | when the mask job exists |
| after its trainer gains the anchor and the worse-of-two identity; a healthy bigbatch 3090 (52, 53, 54, 58, 63 probed empty, 52 and 53 see `/datasets` under Slurm) | R4 | the reward stage from R3's best checkpoint: worse-of-two identity, ImageReward on, anchor, EMA, held-out pairs | picture, claim 5 | after R3 reads |
| now, mscluster107 GPU 1 (RTX 8000, empty on 2026-09-25 although Slurm lists the node down) | R0 | v58-05 again with a different training seed | whether "reliable after N steps" is a claim about the recipe or about one run, claim 1 | whenever a card frees |

Already in hand, no run needed: v58-03 against v58-05 is the contrast-term axis (claim 2.1: the
term costs the cat on cat×dog seed 42 at every checkpoint; v58-05 has the clearest
elephant×penguin at 15k), and run 1 decides whether every sheet from here is read from the EMA
weights. Every run is EMA on, and every run is read by the same judge on the same held-out
sheet, per seed, with the cat×dog seed 42 cell and the striped tiger×dog seed 42 cell watched by
eye because the judge is a coin flip there.

## What the words are

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

| My phrase | The field's name | What it means | Confidence |
|---|---|---|---|
| training on plain-PoE states, running on corrected ones | exposure bias, or covariate shift in distillation; the fix family is on-policy distillation (DAgger) | the student is trained on states from a different trajectory distribution than the one it visits at test time | confident on the names |
| one object borrows the other's features | subject mixing, attribute leakage | cross-attention for one token spills onto the other object's region | confident; "catastrophic neglect" (Attend-and-Excite) is the missing-object case instead |
| the EMA weights | Polyak (exponential moving) averaging | a running average of the weights that damps late-training oscillation | confident |
| the reward drags the adapter off | reward hacking; fixed by KL-regularised or anchored reward fine-tuning | the optimiser finds images that score well and look wrong | confident on the family |
| the render flips on tiny weight changes | sensitivity near a mode boundary | the sample sits where a small change in the score picks a different outcome | family only; no settled name for the checkpoint version |
| the EMA weights | Polyak averaging (Polyak and Juditsky 1992); the EMA copy is what every released diffusion checkpoint is | the average of the training path lands closer to the optimum than the last iterate | confident |
| the `--adapter` cache | one round of DAgger (Ross, Gordon and Bagnell 2011): states the student visits, labelled by the teacher | trains the adapter on its own trajectory instead of plain PoE's | confident on DAgger; on-policy distillation for diffusion, likely but verify the citation |
| the reward drags the adapter off | reward hacking; KL-anchored reward fine-tuning (DPOK, Fan et al. 2023; the RLHF KL-to-reference recipe) | a penalty on distance from a reference model caps what the reward can buy | confident on the family |
| the penguin-bodied elephant | subject mixing, attribute leakage (Attend-and-Excite, Chefer et al. 2023); the training-time fix is a token-level cross-attention loss with segmentation masks (TokenCompose, Wang et al. 2024) | each subject token's attention is held to its own object's region | confident |
| the checkpoint is a third seed | checkpoint variance, seed-to-seed variance of fine-tuning (Dodge et al. 2020 on BERT) | training randomness and checkpoint choice act as extra seeds a recipe claim must average over | family only for diffusion adapters |
| a stopping step read off pictures | early stopping on a validation metric | the metric has to see species; claim 3 | confident on the name, the metric is open |

## Held claims

Navigation: ⬅️ [What the words are](#what-the-words-are) | 📋 [TOC](#table-of-contents) | [Next](#dead-ends) ➡️

| Claim | What is unresolved | What would settle it |
|---|---|---|

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

The prompt's "Already tried" list is the starting set and is not repeated here: samplers with no
correction, Feynman-Kac steering, render-time fixes, samplers bolted onto v58, training longer,
rank and self-attention, the energy penalty, empty-branch-only adaptation, real photographs, CFG++,
APG, norm keeping, manifold projection, growing the pool, and the three unusable scores. See
[the opening prompt](OPENING_PROMPT.md).

| Claim | The workaround | Why it failed |
|---|---|---|
| 5 | a branch-wise attention loss (each branch's one token held to its concept's mask from the target) as the first fix for hybrids | it constrains where a branch attends, and where is already solved: two bodies appear in 24 of 24 cells. The hybrid failure is content inside a region (tiger texture on the dog's body), which comes from the summed epsilons of both branches over that region, not from attention placement. The map can be made to look right while the epsilon is unchanged. Kept as a later probe, not the first run |

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Claim | The check | What each outcome means |
|---|---|---|
| 1 | render v58-03 @25k to 30k and v58-05 @15k twice from the same seed, then the same with `lora_state_ema` | identical repeats and a stable EMA sheet: swings come from weights, EMA is a free fix. Repeats differ: the render is nondeterministic and every checkpoint comparison so far needs a second look |
| 4 | read the sheets of the four empty-branch runs and the attention-intervention output, both on disk; rerun `--out-of-span-only 1` past step 541 | null-only separates bodies but not species: the species fix is token-level. Out-of-span-only run keeps contrast with composition held: the haze cause is the copied damping |
| 3 | run the per-instance species read on the 30 by-eye-labelled cells | agrees on 27 or more: the judge exists and its bar goes in code. Disagrees mainly on look-alike pairs: the judge covers shape-distinct pairs and the look-alikes stay a by-eye sheet. Disagrees broadly: no automatic judge, stopping is by eye on a fixed sheet |

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#routes) ➡️

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|
| 6 | claim 6 | training runs, `scripts/launch_recipe_run.sh` (copied to `outputs/showcase/cohort5_launch_recipe_run.sh`), v58-05's argv as the base, output under `outputs/showcase/cohort5/`: R1 out-of-span-only on mscluster109 GPU 1 (pid 70927), R2a on-policy cache on mscluster106 GPU 1 (pid 2184890), R2b plain cache on the same 81 cells on mscluster107 GPU 1 (pid 1777598), all started 02:25 UTC. R2a died at start-up (its preview sampler needed cells the on-policy cache lacks); relaunched 04:09 on 106 GPU 1 with the plain cache's held-out cells linked in as `onpolicy_cache/cache/heldout` and previews on bear×boar and beaver×raccoon, training from 04:11; failed attempt kept as `cohort5/R2a.failed` and `logs/R2a-failed-0227.log`. A stale launch at 03:12 started a second R1 on mscluster110 into the same folder; it exited, R1 on 109 continued, and `logs/R1.log` holds both processes' output. R0 was refused by the device guard on 107 and has not run | running 2026-09-25 | |
| 5 | claim 5 | Slurm job 59697, `scripts/attention_probe_read.py` over four captures from `capture_attention.py`: cat×dog seed 9 and tiger×dog seed 42, each with v58-05 @15k attached and plain PoE, the share of each branch token's 32×32 attention inside its own concept's box and the other's, per step | done 2026-09-25, output in `artifacts/results/can-a-score-see-species/attention-probe/` | the attention is uniform. In every capture each token's share inside a box equals the box's share of the frame: cat×dog seed 9 with the adapter, the cat token puts 0.29 in the cat box (box is 0.27 of the frame) and 0.45 in the dog box (0.44 of the frame); the dog token the mirror. Plain PoE and tiger×dog seed 42 read the same, at 0.58 against 0.53 and 0.53 against 0.51. A single-token branch's cross-attention, aggregated over layers at 32×32, carries no spatial preference to constrain, so the branch-wise attention loss has nothing to push on, and the dead end stands. Second finding: GroundingDINO put the "tiger" and "dog" boxes on the same body in three of four images, so per-concept boxes are not the way to build the masks; the instance ("animal") boxes plus the judge's per-body label are |
| 4 | claim 3 | Slurm job 59696, `scripts/species_judge.py` second pass: labels redone at full size from the annotated tiles (19 confident of 29), CLIP ViT-L/14, a three-way choice with "half a and half b" as the third answer, other bodies painted out of each crop | done 2026-09-25, output in `artifacts/results/can-a-score-see-species/run-02/` | agrees on 13 of 19 confident rows (cat×dog 9 of 12, tiger×dog 4 of 7). The six misses are the cells where a small white animal is hard to name even at full size (cat×dog seed 42 at 10k to 30k) and the striped-dog hybrids on tiger×dog, which the judge splits between "hybrid" and "two tigers". Right on every clear cell, a coin flip on the ambiguous ones. Below the 27-of-30 bar; usable to rank checkpoints, not yet to stop a run on its own |
| 3 | claim 4 | read on disk, no GPU: v58-04-freezenull (empty branch frozen, W&B `mwrgppyw`) beside v58-05 on cat×dog and tiger×dog, seeds 9 and 42, checkpoints 10k to 45k, tiles from the ablation grid | done 2026-09-25 | freezing the empty branch does not fix species and costs the picture: cat×dog seed 9 becomes a grey drawn tableau with four or five extra animals; seed 42 does get a small cat beside the dog (where v58-05 draws two dogs) but as a painting; tiger×dog seed 9 draws people in hats with tiger and dog heads plus invented glyphs; seed 42 draws a white lion and a wolf. Together with the recorded "adapting only the empty branch gives one-animal clip art", both branch changes are nulls on species, so the species fix is token-level |
| 2 | claim 3 | Slurm job 59695, `scripts/species_judge.py` on the 29 tiles labelled by eye in `artifacts/results/can-a-score-see-species/labels.json`: instance count, then CLIP over the two concept names per body, hybrid margin 0.10 | done 2026-09-25, output in `artifacts/results/can-a-score-see-species/run-01/` | agrees with the eye on 10 of 23 confident rows (cat×dog 8 of 13, tiger×dog 0 of 6, cat×fox 2 of 4). The annotated tiles show three causes: the thumbnail-scale eye labels were wrong on several (a striped white dog called "composed", a small cat called "dog"), overlapping boxes put the dog's body in the cat's crop so a white cat scores as a dog, and a two-way softmax cannot say hybrid (a striped dog scores 1.0 tiger). Inconclusive: the reference labels need redoing at full size and the crops need masks, not boxes |
| 1 | claim 1 | Slurm job 59686, `scripts/render_repeat_and_ema.sbatch`: v58-05 @15k and v58-03 @25k, 27.5k, 30k, each rendered raw in two separate processes and once from `lora_state_ema`, on five held-out pairs at seeds 9, 10, 42; ends by printing the largest pixel difference per tile pair | running 2026-09-25, first checkpoint's two raw passes done | repeat half, read early on v58-05 @15k: the two separate processes differ by 0 of 255 in every pixel on all 15 cells (five pairs, seeds 9, 10, 42). The render is deterministic on one card, so every checkpoint-to-checkpoint swing seen so far is in the weights. EMA half still rendering |

## Routes

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

| # | Anchor | The ask | Return line | State |
|---|---|---|---|---|

## Sources

Navigation: ⬅️ [Routes](#routes) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Source | What it gives the idea | Confidence |
|---|---|---|
| `plans/09-designing-the-correction-loss/review/06-fitting-the-whole-path-and-handing-back-the-tail.md` | the seed sets engraving against photo on identical weights; haze is a 56% early contrast cut, roughly half objective and half unseen-pair generalisation | the project's own recorded read, four cells |
| `plans/09-designing-the-correction-loss/review/04-freezing-the-empty-branch.md` | the freeze-the-empty-branch comparison is designed and has no verdict yet | the project's own file |
| `/datasets/.../showcase/determinism_repeat_1` and `_2` on the cluster | the rank-8 adapter at 280k, rendered twice on cat×dog seed 9 (λ 0 and 1, early and full windows): all three tile pairs differ by 0 of 255 in every pixel | read 2026-09-25 by pixel diff |
| `artifacts/notes/batch-shape-nondeterminism/` | a fixed batch shape renders bit-identically; a different batch shape changes the unconditional branch by 2e-3 per step and the final latent by up to 1.66 | the project's own measurement, one pair, one seed, one 3090 |

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

Read run 1's EMA table when job 59686 finishes (`open 1`), then read R1, R2a and R2b's first
sheets with the judge and by eye when their first checkpoints land (about a day). Start R0 on
mscluster110 from an interactive shell:
`GPU=0 PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python nohup bash /datasets/mmolefe/poe_repair_min/outputs/showcase/cohort5_launch_recipe_run.sh R0 > /datasets/mmolefe/poe_repair_min/outputs/showcase/cohort5/logs/R0.log 2>&1 &`
