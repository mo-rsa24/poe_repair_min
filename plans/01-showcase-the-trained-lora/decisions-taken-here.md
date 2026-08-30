# Decisions taken here

The ledger of the design walks that minted this scope (diffusion-researcher role, the walks at
`plans/.walk/showcase-the-trained-adapter.md` and `plans/.walk/showcasing-the-trained-lora.md`,
reconciled 2026-08-29). Every choice that could have gone the other way, with its reason, so it
can be reversed by someone who disagrees rather than rediscovered by someone who is confused.

**The adapter under study is `phase1_r8_100k`**: LoRA rank 8, alpha 8, on SDXL cross-attention
q/k/v, 100k optimizer steps (2,000 epochs) over 88 runs (11 pairs x 8 seeds), W&B project
`prime_lab/poe-repair-animals-compose`. The cited cross-seed run `pueuo7bl` is the shelved
cross-seed adapter and serves only as a free replication datum.

## What every figure claims over (population)

Captions claim exactly their tier. Figures on the studied pool say "failing animal pairs, 11
train + 4 held-out". Held-out figures claim animal pairs as a class via the split. SDXL
composition in general is claimed only once the non-animal renders in the cache are scored,
which first requires re-validating the instance-count scorer for scenes that are not two
animals; that re-validation is its own plan. Reverses if: the non-animal scoring lands, at which
point tier three captions open up.

> Held-out means the pairs were never shown during training, so a held-out number says how well
> the adapter does on animals it has not seen.

## The caption the structure figures are entitled to

"Small and shared" is only half-entitled. The measured claim: eight directions fitted on 11
training pairs carry 22.6% of the corrections' energy (norm-matched chance level 16.1%, iid
chance level 2.1%); the same directions carry 7.4% of six unseen pairs' energy (random-direction line k/d =
0.012%); the sharing concentrates in steps 0 to 9 (4.4% at k=8 early, 0.19% late). The honest
phrase is "a rule shared, a vector not": same-pair cross-seed cosine is 0.002, so no single
correction vector exists to memorise. From `outputs/interaction_term/cache_analyses/spectrum.json`
and `spectrum_windowed.json`.

## The figure standard every showcase figure obeys

- A structure figure plots energy-at-k (y, 0 to 1) against k (x): curves train, held-out
  projection, and the applicable chance anchor; self-fit reference levels guard the train curve,
  k/d guards a projection. Never a self-fit reference level against a projection (that mistake
  was made live in a walk and is the reason the rule is written down).
- No MDS or UMAP panel may invite the reader to read distances.
- Dose figures: [compose rate](../../context/world/compose-rate.md) (y, 0 to 1) against lambda
  (x), one curve per injected vector, the
  four-control-row layout, AUC always carrying its meaning in words.
- Directions are indexed by step; a direction found at one timestep is not asserted at another.
- Rank-8 (LoRA weight space) and k=8 (r_t space) are different objects; no figure may imply the
  rank was chosen from the spectrum. SVD on targets, never on LoRA outputs.
- Fidelity and composition are separate axes in every caption; crispness is never evidence about
  generalization in either direction.
- Numbers live in figures and sidecars, not in prose.
- Every caption names its claim tier (population above), its space and metric (protocol below),
  the checkpoint step, and "at guidance 7.5". Mono appears as its measured rate, never an
  assumed 1.0. Thresholds live in code, not prose.

## What a 2D picture may say

Any 2D embedding panel (MDS or PCA of trajectories) claims layout only, never distance. The
concentric-contour cartoon is labelled illustrative wherever it appears, because all starting
seeds share one noise-shell radius and seed character can only be directional. A single run's
x_t trajectory plane is nearly lossless (top-2 variance 99.5% on the run that was measured) but
a plane over r_t or eps_PoE hides roughly 45% of the variance and its caption says so.

## Space, metric and mode: the measurement protocol every figure obeys

Measurement lives in guided-epsilon space (cosine, norm-ratio, per-step curves); illustration
lives in predicted-x0 frames and decoded pixels; every caption carries a "space and metric"
field. The rejected alternative: measuring in x0 space throughout would read more naturally
but silently reweights the timesteps by sigma_t/alpha_t (10.43 at step 2, 0.53 at step 40 on
the run that was measured), making the same adapter look early-dominated purely from the lens.
Perceptual claims (when runs visibly separate, when a sample commits) use decoded-embedding
space: Tweedie-decode, embed, 1 minus cosine on L2-normalised embeddings; DINOv2 when the
claim is about image content, CLIP when the prompt is part of the claim. Raw latent L2 with no
named metric is banned from every figure. Mode rule: learned-vs-actual claims are computed
teacher-forced on cached states; closed-loop curves claim behavior only; sidecars name space
AND mode. Reason: the lambda-0 check that must pass before anything runs reproduced the cached
PoE render's mode but not its pixels with settings matched exactly, so only teacher-forced
comparisons are immune to fp16
drift. No borrowed formula enters without checking it is written for variance-preserving
epsilon-prediction.

> Tweedie-decode means turning a noisy latent into the model's best guess at the clean image in
> one step, using the noise prediction it already computed. AUC, where it appears above, is the
> area under the compose-rate-against-lambda curve, on a 0-to-1 scale: 1.0 would mean every
> generation composed at every lambda, 0.0 that none did.

## The window figure: where the correction acts

All showcase injections run at steps 0 to 10 of the 50-step DDIM schedule, guidance 7.5.
Settled from evidence already on disk (outputs/interaction_term/window/window_curves.json,
8 pairs x 9 windows x 4 seeds, 288 scored runs): compose rate by window centre falls 0.656
(centre 5) to 0.250 (10) to 0.094 (15) to 0.031 (20) to 0.0 (25 on), while visible commitment
sits at steps 18 to 36; late dose tripled changes nothing; dose-matched rescaling keeps the
cliff. The figure is the per-pair overlay: compose rate vs window centre, one curve per pair,
each pair's commitment step marked, showing the intervention door closes at or before visible
commitment and its position is shared while commitment varies. A sharp-edge width-1 series is
owed only if the paper ever claims a door location rather than a door. Captions may say "acts
before the picture decides"; they may not say "because of commitment", and sampler-vs-model
attribution stays in `is-the-gap-the-samplers-or-the-models`. Constraint exported to the
steering work: any residual-free steering must inject before about step 10 to have a chance.
Optional descriptive extra, not owed: the repaired run's appearance strip (decoded predicted-x0
thumbnails at fixed steps, adapter run beside frozen PoE run, closed-loop, labelled
descriptive). Two independent reads support the same door and may join the figure as bands or
an appendix panel: decoded frames visibly separate at steps 13 to 20 (flat before 10;
cache_analyses/trajectory_divergence/divergence.json), and the correction's direction is
coherent early and jittery late (consecutive-step cosine of r_t: 0.564 mean over steps 0 to 9,
-0.174 over steps 20 to 49, measured on cat x dog seed 9; recompute over the 64 showcase runs
if drawn).

## The longer-training question runs as three experiments, re-scoped by evidence

Two walks settled this oppositely and the reconciliation is: all three run. The evidence that
re-scopes them: held-out compose rate saturates by 50k (0.812 at 10k, 0.938 at 20k, 0.961 at
50k and 60k; 70k to 100k were never scored), and F8b puts the adapter at or above what the
cached true correction itself reaches, per pair, on the scored metric. So A and B cannot claim
compose-rate gains; they chase the softness the scorer cannot see, and B's rank-ablation figure.
The free read of where the training curve stops rising runs first and is their interpretation
key: `eval/frac_distance_reached` flattens near 0.4 (the curves from `instrument-02`); flat
across epochs and checkpoints means ceiling, still rising means waypoint.

> The cached true correction is the correction computed from the joined prompt, saved once and
> read back, so it is the best any injected correction could do.

**Experiment A, length.** Resume `phase1_r8_100k` from `lora_step_100000.pt` to 200k steps,
nothing else changed. The null threshold, fixed before launch: held-out compose rate at 200k within the
seed-noise band (spread over the 8 held-out seeds) of 100k, with no crispness change in the
frozen tracking set (defined below), means length is not the knob. Rough cost 6.6 GPU-hours
at the measured 4.2 optimizer steps/s; measure it.

**Experiment B, capacity.** Fresh runs at rank 16 and rank 32 (alpha equal to rank), 100k
steps each, everything else identical (`sweep_s1_rank.sh` is the runner for the capacity axis).
Buys the rank-ablation figure: same pairs, same lambda, same inference across r8/r16/r32 at
matched step counts. The null threshold: r16 and r32 held-out compose rates within r8's
seed-noise band. The panel showing what the cached true correction reaches at best is B's
interpretation key: crisp from that correction while the adapter is soft is the one outcome
that makes capacity the right lever to pull.

**Experiment C, injection.** No training. Lambda grid {0, 0.25, 0.5, 0.75, 1.0} crossed with
the step window at inference on existing checkpoints. If softness tracks lambda at a fixed
checkpoint, the blur is the injection, not undertraining. C is also the intervention that
lets any discovered direction use causal language, and it shares one runner with the
adapter-dose series below.

**Two tasks owed alongside, already routed to figure-01-the-transfer-figures.md:** score the
unscored 70k to 100k per-epoch samples (closing F8a's gap), and the qualitative best-case panel
(adapter-corrected renders beside true-r_t-injected renders, the same held-out pairs and seeds,
caption stating the comparison is qualitative).

**Launch shape.** biggpu, 3-day walltime, one Slurm job per user: A as the sbatch job on an
idle node; B's two runs over the SSH-plus-nohup idle-node path per
`environment/hpc/execution-protocol.md` step 3 (including the admin node-cap fallback:
prefer device 1, verify it is free); C in-session. Roughly 20 GPU-hours across three nodes.
No run launches before its plan file and pre-registered review questions exist, per
`~/.claude/EXPERIMENT_CONVENTIONS.md`. Every launching or harvesting session uses the wandb
MCP and Playwright MCP (registered at user scope; runbook/reading-a-training-run.md section
3) and files captured panels into the places that runbook page keeps for screenshots.

## The shared tracking set extends instrument-02 and nothing else

One tracking set (the fixed list of images and curves saved at every checkpoint), frozen
before launch, rendered every 10k steps, logged to W&B. Contents: the four F9 renders
(held-out pairs, seed 9), the repaired render of cat x dog seed 1, compose rate over the
held-out pool (8 pairs x 8 seeds), direction-cosine and fraction-of-distance-reached (already
wired),
and four additions admitted under one rule (computable from what the tracking set already
renders or records, so training cost does not grow):

1. Learned-vs-actual cosine per timestep bucket: cos(learned delta, cached r_t) at every recorded sampling step (the pooled trainer records all of them), teacher-forced, reported per window bucket per the measurement protocol. Says which part of
   the trajectory is learned first.
2. Embedding drift: DINOv2 and CLIP embeddings of each tracking-set render; distance-to-mono
   minus distance-to-poe against training step, per render. The instance-count scorer keeps
   ownership of composition claims.
3. Spectral share of the learned correction, labelled diagnostic: top-k singular-value share
   of stacked learned deltas at a fixed bucket. The paper's low-rank claim stays on targets
   (r_t), never on LoRA outputs.

> Low-rank means a whole family of corrections can be written as combinations of a handful of
> fixed directions, so a few numbers describe each one.
4. Divergence-step profile: norm of the learned delta across the 50 sampling steps
   (closed-loop by design, labelled so), read against commitment steps 18 to 36 and the best
   injection window 0 to 10.

Reverses if: a proposed addition needs new forward passes, in which case it goes to the
mechanism plan instead of the tracking set.

## The mechanism reads run during training, beside it, never inside it

The learned object first: the repo's own measurement (same-pair cross-seed cosine 0.002) says
r_t's direction is state-specific; the interaction term is a rule, not a vector, so no fixed
injected direction can replace the adapter, and a state-dependent map on cross-attention (the
layer carrying object attribution) is the right kind of object. What runs: h-space (UNet
bottleneck activations, adapter on minus off: the correction in the model's own semantic
space) and Jacobian singular directions (does the adapter open a new high-gain direction for
the missing animal), as one watcher process: it watches the run's checkpoints folder, and
when a broad-interval checkpoint lands (default first, middle, final; adjustable) it computes
both reads on a device that is not the training device and logs to the same W&B run, so the
mechanism figures appear live beside the training curves. The training loop is never
modified; in-loop reads would slow every run and are the rejected alternative. A few showcase
renders; storage forced to pooled activations or a subset of the renders; each read ends in its
intervention (inject the found direction before about step 10, score composition), because a
direction figure without an intervention shows a correlation and nothing more. Full
residual-free steering runs across settings are parked as their own idea thread, not this
paper. Most of the paper's figures measure the cached true correction rather than the adapter,
so this plan is a deliberate scope addition. Reverses if: the interventions fail to move compose
rate, in which case the figures stay out of the main text.

## The dog x dog test, pre-registered

Null-input control: C1 = C2 = "a dog" through PoE + LoRA, window 0-10, existing seeds. Because
the experts agree, the true correction is near zero, and language space agrees (the L1 additivity
gap for an agreeing pair is near zero). Supports the rule story: corrected run still shows one
dog and per-step ||r-hat|| is small on the cross-pair scale. Falsifies it: two dogs (the LoRA
carries a plurality prior; the "learned a rule" caption dies as stated). Inconclusive: one dog
with large ||r-hat||. The identity check, before reading anything: verify PoE(A,A) reduces to
Mono(A) within fp16 drift (copy the `run_cfg_masked(all_off)` identity pattern). Guidance is
never raised to buy crispness, since that pushes the sample off the data manifold.

## The generalization demo ships at the reviewer-credible tier

Single-pair transfer stays a quick wiring check (a memorised correction that happens to fit
cannot be excluded). The demo figure is the transfer matrix for group-pooled training evaluated
on concept-disjoint pairs, with the existing held-out grid as the qualitative view beside it.
The reproducibility diagnostic (two LoRAs on disjoint seed subsets, same eval noise, same
corrected image?) is deliberately optional: one extra training, revisit after the dog x dog test
lands.

## The adapter-dose series is owed and approved

Roughly twenty figures measure the cached r_t; one measured the LoRA. The paper ships the LoRA,
so the causal dose curve must be the LoRA's own: lambda on r-hat with the wrong-seed and shuffled
controls, the same machinery as the cached-correction series with the injection source swapped.
The dog x dog test is this series' zero-interaction control; Experiment C above is its lambda
grid; the three share one runner.

## The joint prompt's own failure: framing and count

The baseline is the target itself: on some held-out runs the joint prompt fails to show both
animals, and the adapter, which never consumes the joint prompt at inference, restores them.
Adopted: (i) score the cached mono.png renders across the held-out pool with the validated
instance-count scorer (a render exists for every pair and seed; in-session, minutes), giving the
joint-prompt baseline compose rate per pair; (ii) the counted figure: compose rate per pair,
three bars (joint prompt, plain PoE, adapter), the same pairs and seeds throughout, with a strip
of repaired renders beside it as the anecdote; (iii) wording rule: "restores composition the
joint prompt loses on this pool", never "outperforms SDXL" or "outperforms large models"; (iv)
guidance stays fixed at 7.5 and stated in every caption; no run across guidance values enters
unless the section order ever claims why the joint prompt mode-drops (it does not today).

## The seed-character panel

A free read from runs already scored: per-seed marginal compose rate across pairs, binomial
noise as the threshold, that threshold written into the source. If some seeds compose across
pairs while others never do, starting directions carry a pair-independent character and the
basin framing gains a measured panel at zero GPU cost; if the spread sits inside binomial noise,
the panel reports that null and the seed-character idea closes.

> A basin is the set of starting noise samples that all end up at the same kind of picture, so
> "which basin a seed is in" is which outcome it was always going to reach.

## The repaired run is a paper fact and a provenance rule

`heldout/a_cat__x__a_dog/seed_1`: the joint prompt renders a single cat, plain PoE renders one
blended creature, and the adapter at lambda 1 from the 100k checkpoint renders a separated dog
and cat plus a third smaller animal a caption must mention. Evidence and card:
`artifacts/results/does-the-fix-reach-unseen-pairs/joint-failure-repair/`. Renders here are
mode-reproducible, not byte-reproducible (fp16), and figure sidecars say so.

## No intrinsic-dimension estimate exists

Said as the role requires: the N >= c*ID placement for the corrector cannot be made, so every
generalization claim rests on the behavioral tests above, not on a data-budget argument. The
missing number has a named, priced fix: a two-NN intrinsic-dimension estimate on the cached
final latents (cache-only, an afternoon, no sampling). No figure waits on it; it
upgrades "2,000 epochs over 88 runs is deep in memorisation territory" from a plausibility
argument to a placement.

## Still open, on purpose

The 50k-versus-100k softness sanity read (free, and nothing waits on it). The outcome of the
panel showing what the cached true correction reaches at best, which decides whether B's result
is read as a capacity finding or a dead lever. The non-animal scorer re-validation, which opens
tier-three captions.
