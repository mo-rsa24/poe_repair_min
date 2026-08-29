# Walk: showcasing the trained LoRA

Role: diffusion-researcher (`~/.claude/roles/diffusion-researcher.md`)
Skill: drip-walkthrough. Compile writes the decision ledger and emits the plan chain.
Target: a family of plans that produce the paper's LoRA-results figures under one shared
figure standard (what is shown, caption simplicity, axes, framing, how a reader traces the claim).

## The feature in one paragraph

The rank-8 cross-attention LoRA (trained on cached r_t = eps_J - eps_PoE, Mono-free at
inference, W&B run prime_lab/poe-repair-cross-seed/pueuo7bl) restores two-instance
composition on held-out pairs but the samples are soft and washed out
(paper/iclr/figures/joint-prompt-poe-and-adapter-samples-for-four-held-out-pairs.png).
The user wants: (a) a decision on whether to train longer, (b) the dog x dog probe (C1 = C2 =
"a dog": does PoE+LoRA paint two dogs, meaning the LoRA carries a two-instance prior, or one
dog, meaning it carries an interaction-specific correction), (c) generalization
demonstrations, (d) a figure standard all resulting figures obey, (e) several plans that
together produce the paper figures.

## Pinned before layer 1 (read from the repo)

- Prediction target: epsilon; targets are guided-epsilon differences r_t = eps_J - eps_PoE.
  VP process, discrete 50-step schedule.
- Space: SDXL latent, 4 x 128 x 128 = 65536 dims per step
  (poe_repair/experiments/compose_scorer_validation/scorer.py).
- Sampler: 50 inference steps, guidance 7.5 (poe_repair/config.py DEFAULT_GUIDANCE).
- LoRA: rank-8 on attn2, per-configuration training; cross-seed pooling holds out seeds
  9-12 (scripts/cross_seed_lora_pooling/heldout_pair.sh); G6 checkpoint currently
  k04__ep2000_resumed, k08__ep1600 named as the per-pair alternative.
- Training-set size vs intrinsic dimension: no intrinsic-dimension estimate exists in the
  repo. The role says this absence is itself the layer-4 finding.

## Layers (role order, fixed)

| # | Layer | Mark | Note |
|---|---|---|---|
| 1 | The structure the data actually has | decided (settled) | spectrum.json: k=8 captures 22.6% of stacked r_t energy vs 2.1% chance floor; a train-fitted subspace captures 7.4% of held-out pairs' energy at k=8: far above the random-direction line (k/65,536 = 0.012%), far below the train curve (22.6%). Windowed: held-out projection at k=8 is 4.4% early (steps 0-9) and 0.19% late (steps 20-49), so the shareable part lives early. Entitled claim: compressible well above chance within the training pairs, weakly shared across pairs. |
| 2 | What the model transports, and between what endpoints | settled: needs-a-decision resolved into a named read | Transport pinned from the repo: input is the PoE run's state through attn2 (Mono-free), output r̂_t, loss L2 on cached r_t logged in three step-windows. The open decision: is the softness a conditional-mean shrinkage (information-limited) or undertraining/capacity? Deciding fact: whether eval/frac_distance_reached's ~40% plateau moves across epochs and checkpoints (already logged; zero new GPU to read). |
| 3 | The trajectory, and what it costs to walk it | decided (settled) | DDIM, 50 steps, deterministic. The window is measured and verdicted: compose rate 0.656 with the injection window at steps 0-10, 0.250 at 5-15, 0.000 from 20-30 on (F4a); tripling late dose changes nothing (F4d); dose-matched rescaling keeps the cliff (F4e); commitment sits at steps 18-36. Every showcase run injects at steps 0-10. Captions may say 'acts before the picture decides'; they may not say 'because of commitment', and sampler-vs-model attribution stays in the is-the-gap-the-samplers-or-the-models scope. |
| 4 | Whether it generalised or memorised | settled: probe approved, demo tier repo-decided | Demo tier is repo-decided: single-pair transfer already downgraded to smoke test, group-pooled disjoint pairs named reviewer-credible (EXP-03 note in context/world/lora-corrector.md). Open: (a) run the dog x dog null-input probe, falsification pre-registered below; (b) optionally the reproducibility diagnostic (two LoRAs, disjoint seed subsets, same eval noise). No intrinsic-dimension estimate exists for the r_t mapping, said as the role requires: the N >= c*ID placement cannot be made. |
| 5 | Reading and steering what it learned | settled: adapter-dose sweep approved | The dial (lambda on the injected correction) and its causal controls are measured and verdicted for the CACHED r_t (dose AUC 0.387 real vs 0.023 random, area under compose-rate-vs-lambda on 0-1). The gap the register already names: ~20 figures measure r_t, one measured the LoRA; the residual-vs-adapter ablation is owed. Decision: run the adapter-dose sweep (lambda on r-hat, wrong-seed + shuffled controls, moderate GPU) vs ship oracle-dose only with an honest caption. Deciding fact: the paper ships the adapter, so the adapter's own dose curve carries the causal claim; recommend run. |
| 6 | Conditioning it on something you actually have | decided (round 1) | The conditioning design is the project's premise: expert prompts only, Mono-free, CFG at 7.5; PoE is a guidance-space vector sum and r_t its missing interaction term. Layer adds one preflight to the dog x dog probe: verify PoE(A,A) reduces to Mono(A) within fp16 drift before reading the LoRA's behavior (the run_cfg_masked all-off identity is the pattern to copy). Guidance stays at 7.5 everywhere; raising it to buy crispness is the role's named trap and is off the table. L1 additivity-gap prediction attached: dog x dog has near-zero gap, so language space predicts no correction needed, a second independent pre-registration for the probe. |

No skips: all six layers have content for this target.

## Vocabulary minted by this walk

| Term | Plain sentence | Where |
|---|---|---|
| heavy tail | the energy left after the best k directions; here 37% remains past k=64, so no small set of template corrections carries the pile | layer 1 expand |
| random-direction line | what a random k-dimensional subspace captures of any vector in expectation, k/65,536; the honest floor for held-out projection | layer 1 expand |
| norm-matched floor | chance recomputed with the real rows' norms kept, the fairer self-fit floor; train beats it modestly (22.6% vs 16.1% at k=8) | layer 1 expand |
| conditional-mean shrinkage | an L2-trained predictor whose inputs underdetermine the target outputs the average of the possibilities; averaging near-orthogonal corrections gives a small, smooth, generic output | layer 2 |
| fraction-of-distance-reached | how far the applied correction moves the run from PoE toward Mono, 0 to 1; logged live as eval/frac_distance_reached, plateaus near 0.4 | layer 2 (instrument-02) |
| energy at k | the fraction of the stacked r_t vectors' total squared length that the best k directions capture; 1.0 would mean k directions carry everything | layer 1 |
| chance floor | the energy-at-k a same-shaped pile of random Gaussian vectors would give, so any claim of structure must beat it | layer 1 |

## Deferred machinery

(none yet)

## Open decisions

- Dog x dog probe, pre-registered before any run (layer 4, round 1): pass C1 = C2 = "a dog"
  through PoE + LoRA, inject at steps 0-10, seeds and guidance as the grid used. The true
  correction is near zero because the experts agree. Supports the rule story: the corrected run
  still shows one dog and per-step ||r-hat|| is small against the cross-pair ||r-hat|| scale.
  Falsifies it: two dogs appear (the LoRA carries a plurality prior, and the "learned a rule"
  caption dies as stated). Inconclusive: one dog but ||r-hat|| large (correction present but
  cancels; needs the direction read). Cheap: hours, existing machinery. Needs its own plan +
  review file per EXPERIMENT_CONVENTIONS before running.

- Figure standard, first entry earned by layer 1: a structure figure plots energy-at-k (y, 0 to 1)
  against k (x), three curves (train, held-out projection, chance floor), from
  outputs/interaction_term/cache_analyses/spectrum.json. No MDS/UMAP panel may invite the reader
  to read distances.
- Caption entitlement: "small and shared" must be qualified; the shareable cross-pair part is
  small (below the chance line at k=64) and the honest phrase is "a rule shared, a vector not".
- Rank sweep and train-longer, now one decision with its fact named (layer 2, round 1): read the
  already-logged curves first: eval/frac_distance_reached and loss(early/commit/late) across epochs
  on existing checkpoints plus the in-flight instrument-02 run. Plateau flat across epochs =>
  conditional-mean account, longer training and rank buy nothing, the fix is richer inputs or an
  honest reframe. Plateau still rising => undertrained, train longer justified; rising with rank
  in sweep_s1_rank.sh => capacity. Read order: logged curves (free) before any GPU-days.
  The instrument plan itself records "the ~40% plateau we see in correction magnitude".
(user asks still queued into layers 4 and 6)

## Routes emitted

- submerge journey planned and committed: `~/goal-setting/learning/spectral-structure-of-the-correction/`
  (14 plans, ~7.5h, teaches the SVD machinery layer 1 leaned on; ledgered in JOURNEYS.md, parent
  poe-composition-diffusion). The walk resumes at layer 2 whenever the reader returns; the journey
  runs on the calendar in parallel.
- picture-speak artifact, built and published: "The structure of the correction",
  https://claude.ai/code/artifact/2e8eb1d8-3e6f-4c7c-a3f9-a07d79b14018 (8 guided frames + an
  Explore mode over the real spectra; grounded panels labelled, illustrative panels labelled).

## Next step

Compiled with --write on 2026-08-29. The ledger is the scope's decisions-taken-here.md; the scope is built at plans/01-showcase-the-trained-lora/ (now thirteen plans, twelve review files, illustrated map, root running-order steps 31-43) and verified. This walk is complete.
