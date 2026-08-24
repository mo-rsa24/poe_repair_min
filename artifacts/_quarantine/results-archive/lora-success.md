# LoRA — scaling beyond a single cell

Can the LoRA approach (rank-8 cross-attention adapter that learns the
Mono−PoE residual on cached trajectories) be extended to fix PoE
reliably across many collision pairs, or is it fundamentally a
single-cell trick?

This document writes down the realistic options, what each one buys, and
what each one would cost. It is meant for someone who has seen the
single-cell result and is wondering "now what."

---

## Part 1 — What the single-cell result actually proved

On the cat × dog cell with seed 42, after roughly 600 epochs of training
on the cached trajectory, the LoRA-corrected PoE prediction was about
42% of the way (in pixel-L1) from plain PoE to cached Mono — and still
moving. The visual trajectory across epochs 100, 300, 400, 500, 600 was
a smooth morph from "blob of fur" hybrid, through "cat tucked into dog
chest" cuddling, to "white cat and large white dog clearly visible side
by side, touching." The trajectory had not plateaued by epoch 600.

Three things this proves, in order of how load-bearing they are:

1. **The hypothesis space is right.** A rank-8 LoRA on the SDXL
   cross-attention projections (`attn2.to_q`, `attn2.to_k`, `attn2.to_v`)
   *can* hold a useful correction. We did not run out of LoRA capacity
   on one cell — the limiting factor was training time and the gap
   between cached and rollout trajectories, not the LoRA's expressive
   power.

2. **The training loss is tractable.** The per-arm objective
   (drive `ε_PoE_lora` toward `ε_J_cached`) gives non-degenerate
   gradients from step one. Training is stable, monotonically (and
   slowly) decreasing.

3. **Inference is Mono-free.** The deployment sampler never encodes the
   joint prompt. It only uses the LoRA-modified per-arm forwards on
   "a cat", "a dog", and empty prompt. So the LoRA is the *only* thing
   carrying the correction at deployment, which is the whole point.

What this *did not* prove: anything about pairs other than cat × dog,
seeds other than 42, or any kind of generalization. The LoRA was trained
on one cached trajectory and we measured it on the same trajectory.

---

## Part 2 — Why scaling is genuinely hard

Three obstacles stand between "fix one cell" and "fix many cells":

### Obstacle A — Negative transfer

If you train a single LoRA on cached trajectories from many different
pairs simultaneously, the LoRA has to find a single set of cross-attn
weight perturbations that helps every pair. Different pairs sometimes
want contradictory corrections at the same attention layer. Cat × dog
might want the cat-token to spread spatially; bird × frog might want it
to concentrate. One LoRA averages those, helps neither.

This is the dominant failure mode of "just train on more data."

### Obstacle B — Capacity

Rank-8 was enough for one cell because we used essentially all of it.
Asking one rank-8 LoRA to encode the right correction for ten different
pairs is asking too much of 5 million parameters. The fix is more rank
(32 or 64), which is still small relative to SDXL's 2.6 billion
parameters, but it does not magically resolve negative transfer.

### Obstacle C — Rollout drift

This is the subtle one and it affects every road below.

The training data is cached at the *PoE trajectory's* z_t at each step.
But at inference, when we apply the LoRA correction with non-zero λ,
the trajectory drifts away from PoE's path — it heads toward something
between PoE and Mono. The LoRA was never trained on those drifted z_t.
By the end of denoising, the latent the LoRA sees during inference can
be meaningfully different from the latent it saw during training, and
its prediction quality degrades.

This is plausibly why we asymptote around 42% toward Mono on cat × dog
instead of reaching Mono identically. Across many pairs this gap likely
compounds.

---

## Part 3 — Three roads to multi-cell repair

### Road 1 — A library of per-cell LoRAs

**What it is.** Train one separate LoRA per (pair, seed) you care
about. Save each as a 20 MB checkpoint. At inference, look up the right
LoRA by the prompt pair and load it.

**Why it works.** It is exactly what we just did, repeated. No new
research, no shared-weight contention. Each LoRA is optimal for its
cell. We already know this gives a non-trivial improvement on cat × dog
seed 42.

**What it costs.** Each cell needs roughly six to nine hours on a single
GPU at the speeds we measured. To cover, say, 50 collision pairs at 3
seeds each (150 cells), that is ~1000 GPU-hours, which is small on a
cluster.

**What it doesn't do.** It cannot generalize to a pair you didn't
train on. If a user comes along with "lighthouse × pizza," you have no
LoRA for them.

**When to choose it.** If you have a known, finite set of collision
pairs that matter (a benchmark, a product feature, a study), this is
boringly reliable. Storage is negligible. Lookup is trivial.

### Road 2 — One shared LoRA across many cells

**What it is.** Build a training set that mixes cached trajectories
from many pairs. Train one rank-32 LoRA on all of them. Loss is
averaged across pairs.

**Why it might work.** Cross-attention is already prompt-conditional by
construction — the keys and values come from the text embedding of the
prompt being processed. A LoRA that perturbs `to_k` and `to_v`
therefore produces a perturbation that automatically depends on the
prompt. So the same LoRA weights produce different effective
corrections for different prompts, without us having to engineer that
behavior. This is the property that *might* let one LoRA cover several
pairs.

**Why it might fail.** Negative transfer (Obstacle A above). If two
pairs disagree about what the cross-attn perturbation should look like
on some shared geometry, the averaged gradient pulls toward neither.

**Practical experiment shape.** Curate 5 to 20 pairs you care about.
Cache their trajectories. Train a single rank-32 LoRA for, say, 500
epochs. Evaluate it per pair against the same pair's cached Mono. The
per-pair scores tell you whether you got coverage or paid for the
sharing in degraded per-cell performance.

**Expected outcome, honest guess.** Covers a handful of related pairs
well. Degrades when the set spans qualitatively different failure
modes. Does not handle out-of-distribution pairs.

### Road 3 — Hypernet-conditioned LoRAs (mixture of LoRAs)

**What it is.** Don't train a fixed LoRA. Train a small *hypernet*
(another neural network) that, given the prompt-pair embedding,
*outputs* the LoRA's A and B matrices. The effective adapter at
inference is then a function of which pair you're composing.

**Why it might work.** This is the natural answer to negative transfer.
The hypernet can output a *different* perturbation for each pair
without two pairs fighting over the same weights. There is real
research in this area — Mixture-of-LoRAs, conditional adapters,
prompt-conditioned hypernets — though I am not aware of published work
applying it specifically to PoE repair.

**Why it might fail.** The hypernet has to learn a generalizing
mapping from prompt to correction. It might just memorize the training
pairs and produce nonsense for unseen ones. Distinguishing "learned a
real mapping" from "learned a lookup table" requires careful held-out
evaluation.

**Costs.** Total trainable parameters jump to 30–100M (hypernet +
adapter spec). Training takes longer (more data, slower convergence).
Probably weeks of work to set up, run, and evaluate.

**Expected outcome, honest guess.** Best path to "reliable across many
pairs." But the project size is multiple weeks, not days. And whether
it generalizes to genuinely unseen pairs (not in the training
distribution at all) is the empirical question that needs to be
answered, not predicted.

---

## Part 4 — The lurking problem: rollout drift, and how to fix it

All three roads above inherit the same fundamental issue: the loss
function (match cached Mono on cached PoE trajectory) doesn't directly
optimize what we want (produce a clean cat-and-dog image at deployment).
This is the rollout-drift gap.

The standard fix is **outcome supervision**: instead of matching cached
tensors, optimize through a downstream grader. Three known approaches
that apply here:

- **DRaFT** — backpropagate gradients all the way through the sampler
  and a differentiable grader (e.g. a CLIP-based or VQA-based scorer).
  Most directly addresses rollout drift because gradients see the actual
  inference trajectory.
- **AlignProp** — similar in spirit, with some practical refinements.
- **DDPO** — reinforcement learning style; works with a black-box reward
  (e.g. GroundingDINO's `both_distinct` regime classification). Doesn't
  need a differentiable grader but is sample-inefficient.

Combined with any of the three roads above:

- Library of per-cell LoRAs + outcome supervision: each cell-specific
  LoRA gets closer to actual Mono behavior than cached MSE allows.
- Shared LoRA + outcome supervision: the shared LoRA is optimized for
  the property we care about (visible co-occurrence) rather than for
  matching a cached number that may not correspond to the property.
- Hypernet LoRA + outcome supervision: the hypernet learns to output
  LoRAs that *work at inference*, not LoRAs that *match cached
  residuals*.

Outcome supervision is the second lever, and it's largely orthogonal to
the architectural choice (Road 1, 2, or 3).

---

## Part 5 — Verdict and recommendation

The honest summary:

| Question | Answer |
|---|---|
| Can rank-8 LoRA fix one (pair, seed)? | Yes — proven on cat × dog seed 42. |
| Can a library of per-cell LoRAs fix any cell you cache? | Yes, with linear effort. |
| Can a single rank-32 LoRA cover ~10 curated pairs? | Probably yes, with some per-cell degradation. Worth running. |
| Can a single LoRA cover all collision pairs out of the box? | No. Capacity and negative transfer kill it. |
| Can a hypernet-conditioned LoRA + outcome supervision cover many pairs reliably? | Plausibly yes. Real research project. |
| Can this generalize to genuinely unseen pairs? | Unknown. Empirical question. |

So: not a hard no, but not a quick yes either. The single-cell result
proves the mechanism is right, which is real evidence. Scaling is a
clear research direction with two named levers — prompt-conditional
LoRAs (mixture-of-LoRAs / hypernet style) and outcome supervision — that
have not been combined for PoE repair in the published literature I am
aware of. Either of those would be a genuine contribution. Both together
would be ambitious but defensible.

What I'd actually recommend, given limited time:

1. **First**: do Road 1 (library of per-cell LoRAs) on a small set of
   five or six known-failing pairs. This is cheap, gives you a working
   product for those pairs, and tells you whether the single-cell story
   replicates across pairs at all. If even Road 1 fails on most pairs,
   the mechanism is more cell-specific than we currently think and the
   bigger projects aren't worth running.

2. **If Road 1 works on most cells you try**: do Road 2 (shared rank-32
   LoRA on those same cells). Compare per-cell performance to the
   per-cell LoRAs. If shared LoRA gets within 80% of per-cell LoRA on
   most cells, you have a real path to "one model covers many pairs."
   If it's at 30% of per-cell performance, negative transfer is biting
   and you'd need to either curate more carefully or move to Road 3.

3. **Only if both above work, and only if you have multiple weeks**:
   layer outcome supervision (DRaFT or DDPO) on top, and/or move to Road
   3 (hypernet). At that point you're doing publishable research, not
   just engineering.

The single-cell result is a real foothold. Everything above is a credible
path forward. None of it is guaranteed, and the most ambitious version
is a multi-week research project rather than a weekend hack.
