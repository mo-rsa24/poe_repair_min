# Decisions taken here: showcase the trained adapter

Every choice below could have gone the other way. Each carries its reason and what would
reverse it, so a disagreement reverses a decision instead of rediscovering it. Two design walks
produced this ledger (diffusion-researcher role, `plans/.walk/showcase-the-trained-adapter.md`
and `plans/.walk/showcasing-the-trained-lora.md`, reconciled 2026-08-29). The adapter under
study is `phase1_r8_100k`: LoRA rank 8 on SDXL cross-attention q/k/v, 100k optimizer steps over
88 cells, W&B project prime_lab/poe-repair-animals-compose. The cited cross-seed run pueuo7bl
is the shelved cross-seed adapter and serves only as a free replication datum.

## What every figure claims over (population)

Captions claim exactly their tier. Figures on the studied pool say "failing animal pairs, 11
train + 4 held-out". Held-out figures claim animal pairs as a class via the split. SDXL
composition in general is claimed only once the non-animal cache cells are scored, which first
requires re-validating the instance-count scorer for scenes that are not two animals; that
re-validation is its own plan. Reverses if: the non-animal scoring lands, at which point tier
three captions open up.

## What a 2D picture may say

Any 2D embedding panel (MDS or UMAP or PCA of trajectories) claims layout only and never
invites the reader to read distances. The concentric-contour cartoon is labelled illustrative
wherever it appears, because all starting seeds share one noise-shell radius and seed character
can only be directional. A per-cell x_t trajectory plane is nearly lossless (top-2 variance
99.5% on the measured cell) but a plane over r_t or eps_PoE hides roughly 45% of the variance
and its caption says so.

## Space, metric and mode: the measurement protocol every figure obeys

Measurement lives in guided-epsilon space (cosine, norm-ratio, per-step curves); illustration
lives in predicted-x0 frames and decoded pixels; every caption carries a "space and metric"
field. The rejected alternative: measuring in x0 space throughout would read more naturally
but silently reweights the timesteps by sigma_t/alpha_t (10.43 at step 2, 0.53 at step 40 on
the measured cell), making the same adapter look early-dominated purely from the lens.
Perceptual claims (when runs visibly separate, when a sample commits) use decoded-embedding
space: Tweedie-decode, embed, 1 minus cosine on L2-normalised embeddings; DINOv2 when the
claim is about image content, CLIP when the prompt is part of the claim. Raw latent L2 with no
named metric is banned from every figure. Mode rule: learned-vs-actual claims are computed
teacher-forced on cached states; closed-loop curves claim behavior only; sidecars name space
AND mode. Reason: the lambda-0 canary reproduced the cached PoE render's mode but not its
pixels with settings matched exactly, so only teacher-forced comparisons are immune to fp16
drift. No borrowed formula enters without checking it is written for variance-preserving
epsilon-prediction.

## The figure standard every showcase figure obeys

- A structure figure plots energy-at-k (y, 0 to 1) against k (x): curves for train, held-out
  projection, and the applicable chance anchor; self-fit floors guard the train curve, k/d
  guards a projection. Never a self-fit floor against a projection.
- Dose figures: compose rate (y, 0 to 1) against lambda (x), one curve per injected vector,
  the four-control-row layout, AUC always carrying its meaning in words.
- Directions are indexed by step; a direction found at one timestep is not asserted at
  another.
- Rank-8 (LoRA weight space) and k=8 (r_t space) are different objects; no figure may imply
  the rank was chosen from the spectrum. SVD on targets, never on LoRA outputs.
- Fidelity and composition are separate axes in every caption; crispness is never evidence
  about generalization in either direction.
- Numbers live in figures and sidecars, not in prose.

## What the structure figures are entitled to claim

"Small and shared" is only half-entitled. The measured claim: eight directions fitted on 11
training pairs carry 22.6% of the corrections' energy (norm-matched chance floor 16.1%, iid
floor 2.1%); the same directions carry 7.4% of six unseen pairs' energy (random-direction line
k/d = 0.012%); the sharing concentrates in steps 0 to 9 (4.4% at k=8 early, 0.19% late). The
honest phrase is "a rule shared, a vector not": same-pair cross-seed cosine is 0.002, so no
single correction vector exists to memorise. From
`outputs/interaction_term/cache_analyses/spectrum.json` and `spectrum_windowed.json`.

## The window: where the correction acts, and the caption boundary

All showcase injections run at steps 0 to 10 of the 50-step DDIM schedule, guidance 7.5.
Measured (outputs/interaction_term/window/window_curves.json, 8 pairs x 9 windows x 4 seeds,
288 scored cells): compose rate 0.656 (window 0-10) falls to 0.250 (5-15), 0.094, 0.031, 0.0
(20-30 on); late dose tripled changes nothing; dose-matched rescaling keeps the cliff. Visible
commitment sits at steps 18 to 36. The settled figure is the per-pair overlay: compose rate vs
window centre, one curve per pair, each pair's commitment step marked, showing the door's
position is shared while commitment varies. Captions may say "acts before the picture
decides"; they may not say "because of commitment", and sampler-vs-model attribution stays in
`plans/is-the-gap-the-samplers-or-the-models/`. A sharp-edge width-1 sweep is owed only if the
paper ever claims a door location rather than a door. Optional descriptive extra: the repair
cell's appearance strip (decoded predicted-x0 thumbnails at fixed steps, adapter run beside
frozen PoE run, closed-loop, labelled descriptive). Constraint exported to the steering work:
any residual-free steering must inject before about step 10.

## The longer-training question runs as three experiments, re-scoped by evidence

The user reconciled the two walks' opposite settlements as: all three run. The evidence that
re-scopes them: held-out compose rate saturates by 50k (0.812 at 10k, 0.938 at 20k, 0.961 at
50k and 60k; 70k to 100k were never scored), and F8b puts the adapter at or above the oracle
ceiling per pair on the scored metric. So A and B cannot claim compose-rate gains; they chase
the scorer-invisible softness and B's rank-ablation figure. The mechanical account of that
softness, kept because it predicts what the runs can find: an L2 loss on a state-specific,
heavy-tailed target commits the adapter to the conditional mean of what its inputs pin down;
averaged near-orthogonal corrections are small and smooth, which accounts for the washed-out
held-out column. Its logged signature: `eval/frac_distance_reached` plateaus near 0.4.

**The plateau read runs first and informs, but does not gate.** Read
`eval/frac_distance_reached` across epochs and checkpoints: flat across epochs reads as
ceiling (A and B buy nothing on this axis; the moves are richer corrector inputs or an honest
reframe), still rising reads as waypoint. Either verdict re-scopes how A's and B's results are
read; neither stops their launch, by the user's explicit call.

**Experiment A, length.** Resume `phase1_r8_100k` from `lora_step_100000.pt` to 200k steps,
nothing else changed. Null bar, fixed before launch: held-out compose rate at 200k within the
seed-noise band (spread over the 8 held-out seeds) of 100k, with no crispness change in the
frozen tracking set (defined below), means length is not the knob. Rough cost 6.6 GPU-hours at
the measured 4.2 optimizer steps/s; measure it.

**Experiment B, capacity.** Fresh runs at rank 16 and rank 32 (alpha equal to rank), 100k
steps each, everything else identical. Buys the rank-ablation figure: same cells, same lambda,
same inference across r8/r16/r32 at matched step counts. Null bar: r16 and r32 held-out
compose rates within r8's seed-noise band. The oracle-ceiling panel (below) is B's
interpretation key: oracle crisp with adapter soft is the one outcome that makes capacity the
regime-correct lever.

**Experiment C, injection.** No training. Lambda grid {0, 0.25, 0.5, 0.75, 1.0} crossed with
the step window at inference on existing checkpoints. If softness tracks lambda at a fixed
checkpoint, the blur is the injection, not undertraining.

**Two tasks owed alongside, already routed to figure-01-the-transfer-figures.md (reconcile,
never duplicate):** score the unscored 70k to 100k per-epoch samples (closing F8a's gap), and
the qualitative oracle-ceiling panel (adapter-corrected beside true-r_t-injected renders, same
held-out cells, caption stating the comparison is qualitative).

**Launch shape.** biggpu, 3-day walltime, one Slurm job per user: A as the sbatch job on an
idle node; B's two runs over the SSH-plus-nohup idle-node path per
`environment/hpc/execution-protocol.md` step 3 (including the admin node-cap fallback: prefer
device 1, verify it is free); C in-session. Roughly 20 GPU-hours across three nodes. No run
launches before its plan file and pre-registered review questions exist, per
`~/.claude/EXPERIMENT_CONVENTIONS.md`. Every launching or harvesting session uses the wandb
MCP and Playwright MCP (registered at user scope; runbook/reading-a-training-run.md section 3)
and files captured panels into that runbook page's screenshot slots.

## The shared instrument extends instrument-02 and nothing else

One tracking set (the fixed list of images and curves saved at every checkpoint), frozen
before launch, rendered every 10k steps, logged to W&B. Contents: the four F9 cells (held-out
pairs, seed 9), the repair cell cat x dog seed 1, compose rate over the held-out pool (8 pairs
x 8 seeds), direction-cosine and fraction-of-distance-reached (already wired), and four
additions admitted under one rule (computable from what the tracking set already renders or
records, so training cost does not grow):

1. Learned-vs-actual cosine per timestep bucket: cos(learned delta, cached r_t) at the
   recorded steps 7, 15, 22, teacher-forced per the measurement protocol. Says which part of
   the trajectory is learned first.
2. Embedding drift: DINOv2 and CLIP embeddings of each tracking-set render; distance-to-mono
   minus distance-to-poe against training step, per cell. The instance-count scorer keeps
   ownership of composition claims.
3. Spectral share of the learned correction, labelled diagnostic: top-k singular-value share
   of stacked learned deltas at a fixed bucket, per the SVD-on-targets rule above.
4. Divergence-step profile: norm of the learned delta across the 50 sampling steps
   (closed-loop by design, labelled so), read against commitment steps 18 to 36 and the best
   injection window 0 to 10.

Reverses if: a proposed addition needs new forward passes, in which case it goes to the
mechanism plan instead of the tracking set.

## The mechanism reads run during training, beside it, never inside it

The learned object first: r_t's direction is state-specific (cross-seed cosine 0.002); the
interaction term is a rule, not a vector, so no fixed injected direction can replace the
adapter, and a state-dependent map on cross-attention (the layer carrying object attribution)
is the right kind of object. What runs: h-space (UNet bottleneck activations, adapter on minus
off: the correction in the model's own semantic space) and Jacobian singular directions (does
the adapter open a new high-gain direction for the missing animal), as one follower process:
it watches the run's checkpoints folder, and when a broad-interval checkpoint lands (default
first, middle, final; adjustable) it computes both reads on a device that is not the training
device and logs to the same W&B run, so the mechanism figures appear live beside the training
curves. The training loop is never modified; in-loop reads would slow every run and are the
rejected alternative. A few showcase cells; storage forced to pooled activations or a cell
subset; each read ends in its intervention (inject the found direction before about step 10,
score composition), because a direction figure without an intervention is the
correlational-picture trap. Full residual-free steering sweeps are parked as their own idea
thread, not this paper. Reverses if: the interventions fail to move compose rate, in which
case the figures stay out of the main text.

## The adapter-dose sweep is owed and approved

Roughly twenty figures measure the cached r_t; one measured the LoRA. The paper ships the
adapter, so the causal dose curve must be the adapter's own: lambda on the learned correction
with the wrong-seed and shuffled controls, same machinery as the oracle sweep with the
injection source swapped, AUC beside the oracle sweep's 0.387-vs-0.023 for comparison. The
dog x dog probe is this sweep's zero-interaction control cell; the two share one harness, and
that harness is experiment C's.

## The dog x dog probe, pre-registered

Null-input control: both prompts "a dog" through PoE plus the adapter, window 0 to 10,
existing seeds. Because the experts agree, the true correction is near zero, and language
space agrees (the L1 additivity gap for an agreeing pair is near zero). Supports the rule
story: the corrected run still shows one dog and per-step norm of the learned correction is
small on the cross-pair scale. Falsifies it: two dogs (the adapter carries a plurality prior;
the "learned a rule" caption dies as stated). Inconclusive: one dog with a large learned
correction. Preflight before reading anything: verify PoE(A,A) reduces to Mono(A) within fp16
drift. Guidance is never raised to buy crispness.

## The generalization demo ships at the reviewer-credible tier

Single-pair transfer stays a smoke test (a memorised correction that happens to fit cannot be
excluded). The demo figure is the transfer matrix for group-pooled training evaluated on
concept-disjoint pairs, with the existing held-out grid as its qualitative rung. The
reproducibility diagnostic (two adapters on disjoint seed subsets, same eval noise, same
corrected image?) is deliberately optional: one extra training, revisit after the probe lands.

## The joint prompt's own failure: framing and count

The baseline is the target itself: on some held-out cells the joint prompt fails to show both
animals, and the adapter, which never consumes the joint prompt at inference, restores them.
Adopted: (i) score the cached mono.png renders across the held-out pool with the validated
instance-count scorer (renders exist per cell; in-session, minutes), giving the joint-prompt
baseline compose rate per pair; (ii) the counted figure: compose rate per pair, three bars
(joint prompt, plain PoE, adapter), same cells and seeds, with the repair-cell strip beside it
as the anecdote; (iii) wording rule: "restores composition the joint prompt loses on this
pool", never "outperforms SDXL" or "outperforms large models"; (iv) guidance stays fixed at
7.5 and stated in every caption; no guidance sweep enters unless the spine ever claims why the
joint prompt mode-drops (it does not today).

## The repair cell is a paper fact and a provenance rule

`heldout/a_cat__x__a_dog/seed_1`: the joint prompt renders a single cat, plain PoE renders one
blended creature, and the adapter at lambda 1 from the 100k checkpoint renders a separated dog
and cat plus a third smaller animal a caption must mention. Evidence and card:
`artifacts/results/does-the-fix-reach-unseen-pairs/joint-failure-repair/`. Renders here are
mode-reproducible, not byte-reproducible (fp16), and figure sidecars say so.

## No intrinsic-dimension estimate exists

The N >= c*ID placement for the adapter cannot be made, so every generalization claim rests on
the behavioral tests above (the probe, the transfer matrix, the held-out split), never on a
data-budget argument.

## Still open, on purpose

The plateau verdict (ceiling or waypoint), which re-scopes how A's and B's results read. The
oracle-ceiling panel's outcome, which decides whether B's result is a capacity finding or a
dead lever. The non-animal scorer re-validation, which opens tier-three captions. The
reproducibility diagnostic, optional until the probe lands.
