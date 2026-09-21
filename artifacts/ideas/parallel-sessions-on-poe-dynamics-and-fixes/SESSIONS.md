# Eight sessions, one read-out

Eight Claude sessions, each owning one experiment. Every session designs, plans and runs its own
piece, and every piece reports through the same read-out so the results can sit in one table
afterwards. Two sessions need no GPU and should start first, because one of them decides whether
three of the others can win at all.

## Table of contents

- [How to launch a session](#how-to-launch-a-session)
- [What every session shares](#what-every-session-shares)
- [Order and dependencies](#order-and-dependencies)
- [Session A: the dynamics, off the cache](#session-a-the-dynamics-off-the-cache)
- [Session B: correct early, then clean up](#session-b-correct-early-then-clean-up)
- [Session C: the Langevin corrector at low noise](#session-c-the-langevin-corrector-at-low-noise)
- [Session D: Feynman-Kac steering with the scorer as reward](#session-d-feynman-kac-steering-with-the-scorer-as-reward)
- [Session E: search over the noise](#session-e-search-over-the-noise)
- [Session F: Diffusion Tree Sampling on the corrected proposal](#session-f-diffusion-tree-sampling-on-the-corrected-proposal)
- [Session G: the second pair](#session-g-the-second-pair)
- [Session H: the two reads](#session-h-the-two-reads)
- [What was easy to miss](#what-was-easy-to-miss)

## How to launch a session

Open a fresh Claude session in this repo and paste one line. The letter is the only thing that
changes.

```
Read artifacts/ideas/parallel-sessions-on-poe-dynamics-and-fixes/SESSIONS.md in full. You are
session A. Follow "What every session shares" and then "Session A" to completion. Stay inside
session A's prompt; the other sessions are running elsewhere.
```

Two ways to run it, and they deliver different things.

**Attended.** You watch, and the skills the prompt names (`/integrate-plans`, `/unpack-paper`,
`/drip-execute-plan`) ask you their questions and wait for `compile`. This is the mode the plan
tree's conventions assume. A GPU session will end its turn once its long run is launched under
`nohup`; you reopen it later and say `harvest`.

**Unattended.** Run `/unattended` in a throwaway session with the launch line above as the target,
and it prints a tmux command that answers every question with the session's own recommendation
and records the choice. Design decisions then get made without you. That is acceptable for
sessions A, E task 1, G and H, which are reads and measurements over existing data. For B, C, D
and F, which write new samplers and claim GPUs, run the design step attended and only the execute
step unattended.

## What every session shares

Paste this block at the top of every session prompt. It is the contract that makes eight results
comparable.

```
Shared contract for this session.

Read first, in this order: CLAUDE.md, environment/overview.md, context/00-INDEX.md,
report/00-INDEX.md, then the two findings
report/when-does-the-outcome-lock-in/where-does-each-condition-land.md and
report/does-the-fix-reach-unseen-pairs/is-the-held-out-gap-a-fit-a-drift-or-a-pair-problem.md,
and the scope you are landing in (named below). Follow ~/.claude/EXPERIMENT_CONVENTIONS.md and
~/.claude/RESEARCH_PRACTICE.md.

The read-out every session produces:
- One triptych sheet per condition: rows are the held-out seeds 9 to 16 of cat x dog, columns are
  Mono (the joint prompt "a cat and a dog"), plain PoE, and the solution this session tests. Where
  the solution is built on the adapter, the adapter is rank 32, step 30050, at
  /datasets/mmolefe/poe_repair_min/outputs/showcase/phase1_r32_100k/checkpoints/lora_step_030050.pt,
  at lambda 1.2, and a fourth column shows adapter-alone so the session's addition is visible.
- Every render shares the seed's cached initial noise, 50 DDIM steps, guidance 7.5, 1024 square,
  unless the experiment varies one of those, in which case the varied one is named on the sheet.
- Every image is scored by the validated instance-count compose scorer (context/world/compose-rate.md).
  Compose rate per condition, over the 8 seeds, in a .json sidecar beside the sheet.
- One composing control pair, a_butterfly__x__a_flower_meadow, on the same seeds, so a method that
  breaks what already works is caught.
- Logged to W&B project prime_lab/poe-repair-animals-compose: the sheet as an image, the sidecar as
  an artifact, the run id written into the review file. W&B owns the numbers, the plan tree owns
  the verdict.
- Outputs under /datasets/mmolefe/poe_repair_min/outputs/ only, never /home-mscluster.

Before anything runs: write the falsification criterion with its number (support, null,
inconclusive) into the plan's review file, and put the thresholds in source as named constants.
Integrate the plan into the tree with /integrate-plans before executing; do not run from chat.

Execution: biggpu allows one Slurm job per user, so long runs start with nohup on a free device
and squeue is blind to them. Check torch.cuda.is_available() on the pinned device before real work
(known failure poe-launch-002: a Blackwell card can list in nvidia-smi and still run on CPU).
co3 python at /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python, co3_bw on mscluster110 to 112.
Other sessions are running at the same time; record node, device and PID in the review file and
check pgrep -af 'sweep|train|corrector' before claiming a device.

Shared files: all sessions edit one working tree. Write only your own plan file, review file,
scripts, results folder and finding. Do not edit the root MASTER_PLAN.md running order, a scope
MASTER_PLAN.md plans table, or the reading register's existing rows; append a line to
artifacts/ideas/parallel-sessions-on-poe-dynamics-and-fixes/PENDING_SYNC.md naming what needs
adding, and one /sync-plan-tree run at the end folds them in. Commit nothing; the user commits.

If a skill you invoke asks a question and nobody answers within the turn, take your own
recommendation, write the choice and its reason into the review file, and continue. Do not stall.

A run launched under nohup outlives your turn. End the turn with the exact harvest line (log path,
output path, expected file count) so the next turn, or the user, can pick it up with one paste.
When reopened, harvest first and only then continue.
```

## Order and dependencies

| Session | Needs a GPU | Lands in | Waits on | Start |
|---|---|---|---|---|
| A: dynamics off the cache | decode only | scope 05 | nothing | now |
| B: correct early, then clean up | yes | scope 01 | its own first task is free | now |
| C: Langevin corrector at low noise | yes | scope 06, steps 25 to 27 exist | nothing | now |
| D: FK steering, scorer as reward | yes | scope 06, redesign of step 51 | A's rung 2 to read the result, not to start | now |
| E: search over the noise | first task free, then yes | scope 06 | nothing | now |
| F: Diffusion Tree Sampling | yes | scope 06 | D's reward machinery, A's rung 2 | after D |
| G: the second pair | yes | scopes 04 and 05 | nothing | now, if a device is free |
| H: the two reads | no | scope 06 step 30, standing/literature | nothing | any time |

Session A's rung 2 is the pivot. It says how much of the correction lies inside what the two
experts already span. If most of it does, D, E and F can win and B's schedule is the fix. If most
of it is orthogonal, D, E and F are baselines that fail cleanly and the adapter is the story.

Seven sessions want a GPU. The cluster does not have seven free cards for one user. Start A, E's
first task and H immediately, B and C on the first two free devices, D on the third, and G and F
when something finishes.

## Session A: the dynamics, off the cache

**Direction.** Read the training cache to say where the two experts disagree, how much of the
correction PoE could ever have expressed, and what the adapter actually adds, per step, with no
sampling.

**Produces.** Four figures with sidecars under
`artifacts/results/what-the-correction-is-made-of/`, a finding in `report/when-does-the-outcome-lock-in/`,
a plan and review file in scope 05.

**Prompt.**

```
The block under "What every session shares" applies in full.

Scope: plans/05-when-does-the-outcome-lock-in/. Also read
context/world/interaction-term.md and the memory note on the A/B split in
artifacts/ideas/which-variable-explains-what-poe-is-missing/IDEA_MAP.md claim 2.

The data: /datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/heldout/a_cat__x__a_dog/seed_<n>/residuals/step_<kk>.pt
holds, per step, x_t and the raw predictions eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond in fp16.
Upcast to fp32 before any norm. The on-policy corrected frames are under
/datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/frames/<cond>/seed_<n>/step_<kk>.png.
Read scripts/build_training_cache.py for the PoE rule the cache used, and write the rule down in
the plan before computing anything against it.

Build four rungs, in this order, each with a picture and a number:

Rung 2 first, because it decides other sessions. Project r_t = eps_j - eps_PoE onto the span of
{eps_a - eps_uncond, eps_b - eps_uncond, eps_uncond} per step. Report the in-span share and the
orthogonal share of ||r_t||^2 against step, mean and band over seeds 9 to 16. Falsification: if the
orthogonal share over steps 0 to 10 is above 0.5, PoE could not have supplied the correction by
reweighting; below 0.25, a per-step guidance reweighting is a candidate fix; between is
inconclusive. Fix these constants in source before running.

Rung 1. Tweedie x0-hat under each expert, under PoE and under the joint prompt, decoded at steps
0, 2, 5, 10, 20, 30, 50 for seed 15 (the seed whose PoE run changes animal mid-way). A
per-position map of ||x0_a - x0_b|| in latent space per step. Number: the share of latent
positions where both experts move away from unconditional in the same place, per step.

Rung 3. Load the rank-32 adapter and evaluate its output at the cached states. Project it the same
way as rung 2. Report: in-span and orthogonal share of the adapter's output, and the cosine of its
orthogonal part against r_t's orthogonal part, per step. A per-position norm map of the adapter's
output beside r_t's, seed 15, same steps as rung 1.

Rung 4. From the frames' running estimates, the kinetic energy of each condition's x0-hat track
(sum over saved steps of the squared displacement, in DINOv2 space, matching the both-ness plane
already used) and the straight-line displacement as its floor. One point per seed per condition.
Add a which-animal score per step (cosine to the cat centroid minus cosine to the dog centroid)
so the animal flip is a number.

Do not touch scope 06's sampler-share measurement; that is rung 5 and it has its own plans. Write
the finding with the template at report/finding-and-explainer-template.md.
```

## Session B: correct early, then clean up

**Direction.** Find when the softness enters the corrected run, then test whether a correction
confined to the early steps followed by a clean tail gives two animals at plain-PoE sharpness.

**Produces.** A sharpness-against-step figure, a λ-schedule grid, a tail-length cell, a re-noise
cell, and one triptych sheet per condition. Lands in scope 01 beside
[the λ window experiment](../../../plans/01-showcase-the-trained-lora/plans/experiments/07-experiment-c-lambda-window.md).

**Prompt.**

```
The block under "What every session shares" applies in full.

Scope: plans/01-showcase-the-trained-lora/. Read its plans/experiments/07-experiment-c-lambda-window.md
and review/07-experiment-c-lambda-window.md (sharpness did not fall monotonically with lambda:
75.6 at 0, 55.8 at 0.5, 69.0 at 1, Laplacian variance), and the memory note that the windowed
LoRA sampler leaves the adapter attached after its window (cost one re-render on 2026-09-05).
Before any windowed run, prove the adapter is detached after the window: render plain PoE after a
windowed run in the same process and check it is byte-close to the cached poe.png.

Task 1, no GPU: sharpness (Laplacian variance) of the running x0 estimate against step for all
six conditions from /datasets/mmolefe/poe_repair_min/outputs/showcase/where_each_condition_lands/frames/.
One line per seed, thick mean per condition. Falsification, fixed before looking: if the corrected
run's sharpness is within the plain run's seed band at step 20 and below it at step 50, the
softness enters late and a clean tail can fix it; if it is already below the band at step 20, the
softness is in the committed structure and only a re-noise can reach it.

Task 2: the lambda schedule. lambda 1.2 on steps 0 to 10, linear decay to 0 by step 20, plain PoE
after, against the full-window lambda 1.2 and plain PoE, 8 seeds, both pairs. Also a hard cut at
step 10 with no decay, because the corrected run's commit step ranges up to 35 on some seeds and
the decay is there to catch them.

Task 3: the tail length. The best schedule from task 2 with the plain-PoE tail run at 50 and at
200 DDIM steps (timesteps re-spaced so the switch-off lands at the same noise level).

Task 4: re-noise and redenoise. Take the schedule's latent at step 20, add noise back to the
level of step 35, denoise to the end with plain PoE. One cell with the SDXL refiner on the two
single prompts in place of the base tail, if the refiner is already on disk; if it is not, say so
and skip it rather than downloading it.

Read-out per cell: compose rate, mean sharpness, both-ness on the existing cloud axes from
scripts/showcase/where_each_condition_lands_plot.py. Support: a cell that keeps compose rate
within 1 seed of full-window lambda 1.2 and returns sharpness to the plain-PoE band.
```

## Session C: the Langevin corrector at low noise

**Direction.** Execute scope 06's steps 25 to 27 as planned, and add one read the plans do not
have yet: whether corrector steps at low noise sharpen the corrected run's final image.

**Produces.** What those three plans already promise, plus a sharpness column.

**Prompt.**

```
The block under "What every session shares" applies in full.

Scope: plans/06-is-the-gap-the-samplers-or-the-models/. Execute, in order,
plans/tools/02-the-corrector-and-the-step-size-it-runs-at.md,
plans/hypothesis/03-what-is-left-once-the-chain-settles.md and
plans/hypothesis/04-does-the-corrector-compose-in-the-same-window.md with /drip-execute-plan.
These plans are designed and pre-registered; do not redesign them. Read the section "Where the
two errors can be told apart, and where they cannot" in the scope MASTER_PLAN.md before the first
run and keep its limit in every caption.

One addition, via /integrate-plans into plan 04 before it runs: a condition where the corrector
runs only on the last 15 steps on top of the rank-32 lambda 1.2 run, k in {0, 5, 20}, 8 seeds,
scored for compose rate and Laplacian-variance sharpness. Falsification: sharpness rises with k
while compose rate holds within 1 seed, or it does not. That is the fidelity read; the sampler
share read stays as the plans wrote it.
```

## Session D: Feynman-Kac steering with the scorer as reward

**Direction.** Rerun the particle baseline with a reward that cannot memorise: the validated
instance-count scorer on the model's clean-image estimate, following Singhal et al., arXiv
2501.06848 (max potential, resampling at a few steps, four particles, SDXL).

**Produces.** The redesign of step 51 in scope 06, and a corrected compose fraction for the old
run's unshown particles.

**Prompt.**

```
The block under "What every session shares" applies in full.

Scope: plans/06-is-the-gap-the-samplers-or-the-models/. Read
plans/baselines/09-twisted-smc-on-a-learned-joint-vs-poe-twist.md, its review file, and
report/is-the-gap-the-samplers-or-the-models/does-selecting-among-poe-proposals-compose.md
(inconclusive: the twist memorised its 120 training latents; SMC compose 0.0 against control 0.0
and Mono 1.0 at K 4). The package is poe_repair/experiments/twisted_smc/.

Task 1, cheap: score the three unshown particles per cell in the old run at
/datasets/mmolefe/poe_repair_min/outputs/interaction_term/twisted_smc/twist_w64_b16_lr1e-04_s100000_20260905-072331/samples/step_<kkkkkk>/
(the files ending __p0.png to __p3.png; the unsuffixed __smc.png is the shown particle),
so the old compose fraction is over 12 images and not 3. Write it into that finding's Still open.

Task 2: Feynman-Kac steering, arXiv 2501.06848, read in full with /unpack-paper first. Replace the
learned twist with a reward on x0-hat: the instance-count scorer's count clipped at 2, plus a
small aesthetic term only if the paper's SDXL setting used one. Max potential, lambda 10 as the
paper, resampling at 5 evenly spaced steps, plain PoE as the proposal. K in {4, 16}. 8 seeds, both
pairs. Note this paper is not Skreta et al.'s Feynman-Kac correctors (arXiv 2503.02819), which is
step 30; add a register row for 2501.06848 in plans/standing/literature/reading-register.md and
keep the two apart by name.

Falsification, before running: support if K 16 composes at least 0.25 more often than the
unweighted control; null if within 0.10 of it at both K; inconclusive if the reward on x0-hat at
steps 0 to 10 disagrees with the final scorer verdict on more than half the particles (the reward
is blind where the decision is made). Fix the constants in source.

Task 3, only after A's rung 2 is in: read the result against the in-span share. Selection can
only reach what the proposal spans.
```

## Session E: search over the noise

**Direction.** Ma et al., arXiv 2501.09732, frame inference-time compute as a search over the
initial noise with a verifier. The random-search row is free from existing renders. The zero-order
search is the new experiment, and it is a training-free rival to the early-window correction
because the fork step is 1.

**Produces.** A baseline row and one experiment in scope 06.

**Prompt.**

```
The block under "What every session shares" applies in full.

Scope: plans/06-is-the-gap-the-samplers-or-the-models/. Read arXiv 2501.09732 with /unpack-paper.

Task 1, no GPU: random search at N 8 is already on disk. From the contact sheet's sidecar
artifacts/results/where-does-each-condition-land/cat-x-dog-in-dino-space.json and
/datasets/mmolefe/poe_repair_min/outputs/showcase/figure_r32_030050/results.json, write the
best-of-8 compose verdict for plain PoE on cat x dog (expected 0 of 8) and for the corrected run,
as a baseline row with its source named.

Task 2: zero-order search. For each of the 8 seeds, perturb the cached initial noise z by
z' = (z + sigma * u) / sqrt(1 + sigma^2) with u standard normal, sigma in {0.1, 0.3}, 8 candidates
per seed, render plain PoE, score with the instance-count scorer on the final image, keep the best.
Compose rate of the kept image per sigma. Then the same with the scorer read on x0-hat at step 10
instead of the final image, to see whether the verifier works where the decision is made.

Falsification, before running: support if the kept image composes at least 0.25 more often than
the unperturbed seed at either sigma; null if within 0.10 at both. Also report the both-ness of
kept images on the existing cloud axes, so a "compose" that is a fused face is visible.
```

## Session F: Diffusion Tree Sampling on the corrected proposal

**Direction.** Jain et al., arXiv 2506.20701, back up a terminal reward through the denoising
chain and spend rollouts on high-value early branches. On bare PoE it should fail like D. Its use
here is fidelity: PoE plus the rank-32 correction as the proposal, a compose reward plus a
sharpness reward, search among early branches.

**Produces.** One experiment in scope 06, after D.

**Prompt.**

```
The block under "What every session shares" applies in full.

Scope: plans/06-is-the-gap-the-samplers-or-the-models/. Wait for session D's reward-on-x0-hat
code and reuse it. Read arXiv 2506.20701 with /unpack-paper; check whether the authors released
code before writing any.

Two conditions, 8 seeds, both pairs: DTS-star (the greedy variant) on plain PoE, and on PoE plus
the rank-32 lambda 1.2 correction. Terminal reward: instance count clipped at 2, plus Laplacian
variance normalised by the plain-PoE seed mean. Budget: 16 rollouts per seed, branching in steps 0
to 15 only, since the fork step is 1 and the commit step is 10 to 22.

Falsification, before running: on plain PoE, support if compose rate rises by at least 0.25 over
best-of-16 random seeds (from session E's row); on the corrected proposal, support if sharpness
rises to the plain-PoE band while compose rate holds within 1 seed. Null within 0.10 and no
sharpness change. Report the compute per image beside every number, since the paper's claim is
about compute.
```

## Session G: the second pair

**Direction.** Every triptych above is cat × dog. The two findings' Still open lists ask for the
same reads on a pair the adapter never saw, with both-ness as a pre-registered bar this time, and
the fit-by-tier read at rank 32.

**Produces.** The `--pairs` extension of the render script, the both-ness figures on one unseen
pair, the rank-32 tier figure.

**Prompt.**

```
The block under "What every session shares" applies in full.

Scopes: plans/05-when-does-the-outcome-lock-in/ (task "Extend the render script with --pairs" in
plans/figures/06-where-each-condition-lands.md) and plans/04-does-the-fix-reach-unseen-pairs/.

Task 1: pick the unseen pair from the high band of
report/does-the-fix-reach-unseen-pairs/does-interaction-strength-predict-which-pairs-blend.md,
and write the both-ness bar before rendering: support if PoE mean both-ness sits at least 0.15
below the corrected mean and the corrected band overlaps the joint band. Then render the six
conditions and the per-step frames on that pair with
runbook/running-things-on-the-cluster/reproducing-where-each-condition-lands.md, and produce the
same figures.

Task 2: the fit-by-tier figure at rank 32, step 30050, reusing scripts/showcase/fit_cosine_on_cache.py
with the pool configs under fit_cosine_tiers_r8_030000/pool/, so the per-pair-learning result is
known to be about the architecture or about rank 8.
```

## Session H: the two reads

**Direction.** Two papers need full reads and no GPU: Skreta et al.'s Feynman-Kac correctors
(step 30, currently gated on step 26) and the optimal-transport lens from Kim's post.

**Produces.** Register rows, a cost estimate, and one paragraph.

**Prompt.**

```
Read CLAUDE.md, then plans/06-is-the-gap-the-samplers-or-the-models/plans/ideas/07-feynman-kac-correctors.md.
Step 30 waits on step 26, but the read itself does not: run /unpack-paper https://arxiv.org/abs/2503.02819
now, promote its register row in plans/standing/literature/reading-register.md, write the cost
estimate against poe_repair/composers/poe_langevin.py as it will exist after step 25, and leave the
built-or-cited decision blank until step 26 returns. Recheck whether an implementation exists.

Then read https://jiha-kim.github.io/posts/autoregression-vs-diffusion-understanding-sampling-via-optimal-transport/
and write one paragraph for session A's rung 4: what the Benamou-Brenier action measures on a
denoising track, and what it cannot say about composition. File it beside the reading register.
```

## What was easy to miss

**The λ schedule and the tail length** are the actual "correct early" experiment. The chat listed
the diagnostic, the corrector and the re-noise, but the schedule is the cheapest thing that could
work and it sits in session B.

**The adapter stays attached after a windowed run.** Every session that renders plain PoE after a
corrected run in the same process inherits this. Session B proves detachment first; sessions C, D,
E and F render references before attaching the adapter or in another process.

**The second pair.** Without session G, every triptych is one pair, and the both-ness read stays
post-hoc.

**The three unshown particles** from the old twisted-SMC run cost nothing and belong to session D.

**GPU contention.** Seven of eight sessions want a device. Sessions record node, device and PID
before claiming one, and check for each other with pgrep.

**Rung 5 is not a new session.** The sampler-share measurement is scope 06's own plans, executed by
session C.

**Two Feynman-Kac papers.** Singhal et al. (steering by reward, session D) and Skreta et al.
(correctors for products, session H). Same name, different method; the register carries both.
