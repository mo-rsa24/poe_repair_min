# Phase 0 — Consolidated plan: characterise the residual + learn it

> **North star:** Establish what the oracle residual `Δ_t = ε̃_J − ε̃_PoE` *is*
> (Veracity, characterisation thread), and then prove that a deployable model
> can *learn it* on the simplest scale (M2-diagnostic, learning thread), using
> a single shared eval protocol that doesn't lie about chimera failures.

This document supersedes both [veracity-figure-plan.md](veracity-figure-plan.md)
(v3, post-audit, figure-set thread) and
[m2-residual-diagnostic-plan.md](m2-residual-diagnostic-plan.md) (D0→D3 phased
learning plan). The two threads were running in parallel with mismatched eval
metrics; consolidating fixes that drift and makes the dependency between
characterisation and learning explicit.

---

## 1. Unified narrative spine

> Phase 0 asks two coupled questions about the guided PoE→Mono correction
> `Δ_t = ε̃_J − ε̃_PoE`. **(A) Characterisation**: what is Δ_t — does it
> exist as a definitional object, is it composition-specific, does it sit in
> a measurable basin-commit window, does it move PoE→Mono when injected at
> inference, and does the success/failure of injection survive *detection-
> based* (not CLIP/attention) recognition checks? **(B) Learning**: can we
> fit a model whose hypothesis space actually contains Δ_t, scored using the
> same detection/VQA metrics from (A), at the simplest possible scale —
> one pair, one seed — before any cross-seed / cross-pair claim? (A) supplies
> the target *and* the eval protocol that (B) needs. They share data and
> infrastructure; previously they shared neither metric language nor figure
> language.

Out of scope (both threads): sched-M2 deployment polish, learned ê_J (M1 —
already deployed), M2a tuning (structurally falsified, see M2-thread §2),
cross-pair generalisation beyond the cat × dog headline cell, and any cross-
seed claim beyond seeds {4, 42, 123}.

---

## 2. Thread A — Characterisation (Veracity figure set, v3)

Anchors a single coherent argument across seven figures. Designations:

| Tier | Figure | Premise | Status |
|---|---|---|---|
| Preflight | Fig 1 — Existence (PMI) | identity holds in code | code + render done |
| Discriminator hook | App-B′ — Detection failure modes | chimera fools CLIP/attention; detection separates regimes | code done; render pending |
| Reachability supporting | Fig 2 — Anti-corroboration + basin-commit | ‖Δ_t‖ is composition-specific, timed to commit | code + render done |
| **HEADLINE** | **Fig 4 — Sufficiency** | λ-walk flips chimera→co-occurrence, detection-verified | code done; render pending detection install |
| Reachability appendix | App-A — Trajectory dependence | PoE-anchor vs Mono-anchor shape divergence = failure signature | code done; all variance caches present |
| Reachability appendix | App-C — CFG × timestep grid | commit band is composition-anchored, not CFG-anchored | code done; cfg sweeps cached |
| Reachability appendix | App-E — Window-localised injection | temporal locus of effectiveness ≈ peak band | code done; **blocks on 10 window-injection GPU sweeps** |

**Reading order**: Fig 1 → App-B′ → Fig 2 → Fig 4 → App-A → App-C → App-E.
Rationale: App-B′ is promoted to position 2 so the detection discriminator is
earned before any figure depends on it. Fig 4 is the load-bearing headline.

**Honesty constraints** (carried into Thread B as well):
- GroundingDINO confidence is logit-style, not calibrated. State so in every caption.
- Detection thresholds: `box≥0.35`, `text≥0.25`. State in every caption.
- VQAScore reduction = `min` over three grounded questions (chimera-discriminator default).
- App-C commit column comes from measured x̂₀-stability ≤ 0.5, not by eye.
- Whole-image CLIP-Score is a *proxy* in every figure it appears in; never load-bearing.

For the long form of each figure (idiom, source data, implementation notes),
see [veracity-figure-plan.md](veracity-figure-plan.md). That document remains
the figure-set spec; this consolidated plan defers there for figure-level detail.

---

## 3. Thread B — Learning (M2 phased plan, with v3 eval bolted on)

The headline change vs the original M2 plan: **CLIP-on-Tweedie is demoted
to a proxy**. The deciding metric becomes the detection + VQA protocol that
Thread A uses for Fig 4 / App-B′ / App-E. CLIP can't tell a chimera from
clean co-occurrence; we can no longer use it to grade a learned residual.

### Inventory (unchanged from M2-thread §0)

| Method | What it predicts | Status |
|---|---|---|
| M1 (sched-M2, MLP→ê_J) | joint text embedding | **deployed**; val cos 0.997; flips basin |
| M2a (soft prompt + frozen UNet → Δ̂) | soft prompt | **falsified**; structural impossibility (§M2-thread 1) |
| M2b (small CNN, FiLM cond, → Δ̂ direct) | Δ̂ directly | underperforming; single-cell variant not yet tested |
| M5 (SDXL + LoRA, → Δ̂ via output diff) | Δ̂ via UNet output diff | not yet scaffolded; licensed iff M2b fails |

### Phased plan (decision-matrix unchanged; metric updated)

#### Phase 0 — Re-orient (2 pomodoros, no GPU)
Re-read Fig 4, App-B′, App-A. Three sentences: "Δ_t is composition-specific,
timed to commit, sufficient when injected; detection beats CLIP."

#### Phase 1 — D0: M2b single-cell overfit + replay (1 day)
Already kicked off via `python -m poe_repair.experiments.diag0_overfit_replay
--skip-train-2a`. Replay scoring **must use the v3 eval protocol** (see §4)
not raw CLIP. Decision matrix:

| M2b loss | M2b replay (App-B′ regime) | Verdict | Next phase |
|---|---|---|---|
| → 0 | `both_distinct` | recipe works | Phase 3 (cross-seed) |
| → 0 | `both_overlapping` or `single` | eps-MSE lies; CLIP-on-Tweedie was hiding it | Phase 2 (LoRA + outcome loss) |
| flat / high | irrelevant | architecture binding | Phase 2 (architecture) |

**Kill criterion**: M2b loss > 0.1 after 3000 steps on a single cell → abort.

#### Phase 2 — LoRA on SDXL (2–3 days, only if Phase 1 fails)
Inject LoRA into SDXL UNet cross-attn `to_{q,k,v}`, rank 8. Δ̂ = ε_LoRA(joint) −
ε_frozen(A,B,∅). Train on a single cell, MSE against cached Δ_t. **Eval per §4**.

#### Phase 3 — Cross-seed lift (3–5 days, only if Phase 1 or 2 succeeds)
Compute cross-seed direction agreement of cached Δ_t for the headline cell;
if >0.7, train on seed-averaged target and hold out one seed; if <0.5, the
residual is genuinely seed-specific — train per-seed and report the limit.

#### Phase 4 — Cross-pair (open-ended, re-plan after Phase 3)
Not planned in detail until Phase 3 returns.

---

## 4. Shared eval protocol (the consolidation)

Every M2-thread replay/inference figure scores its output with the same three
metrics that Thread A uses for Fig 4 and App-E. **This is the bridge.**

1. **GroundingDINO regime classification** (load-bearing for failure mode):
   - Run `detect_boxes(image, [prompt_a, prompt_b], box=0.35, text=0.25)`.
   - Map to `{both_distinct, both_overlapping, single, none}` via
     [`classify_detection_regime`](poe_repair/experiments/veracity/metrics.py#L660).
   - **A working learned residual must produce `both_distinct` on the headline
     cell.** Anything else is a fail for the M2 thread's deployment claim.

2. **VQAScore (LLaVA-1.5)** (load-bearing for separability):
   - Three questions: "Is there a {a}?", "Is there a {b}?", "Is the {a}
     clearly separate from the {b}?"
   - Report `min(p_yes)` — most conservative; chimera-discriminator default.

3. **CLIP-on-Tweedie** (proxy only, never load-bearing):
   - Keep computing it (cheap, useful for sanity), but the M2 decision matrix
     above keys off detection regime + VQA, not CLIP.

For the cat × dog headline cell, the v3 thresholds for "PoE→Mono basin flip
worked" are:
- detection regime transitions from `single` / `both_overlapping` (PoE λ=0)
  to `both_distinct` (post-injection),
- `min(p_yes) ≥ 0.5` post-injection (vs PoE baseline which is typically <0.3 for
  the "clearly separate" question),
- CLIP delta is reported but does not gate the decision.

When wiring this into `diag0_overfit_replay`, the helper functions live at
[poe_repair/experiments/veracity/metrics.py](poe_repair/experiments/veracity/metrics.py):
`detect_boxes`, `box_iou`, `classify_detection_regime`, `vqascore_yesno`.
Both lazy-load their backbones (GroundingDINO-Tiny ~700MB, LLaVA-1.5-7b ~14GB).

---

## 5. Kill criteria (cross-thread)

| Trigger | Verdict | Action |
|---|---|---|
| Fig 4 detection panel shows no monotone confidence rise across λ | Sufficiency claim fails | Stop the headline narrative; investigate decoding/prompt issues before retrying. |
| App-B′ misses all three seeds at threshold 0.35 | Detection threshold wrong for this regime | Lower to 0.20 with caveat in caption (open item #4 in v3 plan). |
| App-E VQAScore is flat across windows | Either window injection didn't take or VQA prompt is wrong | Inspect a single decoded window image manually before re-running. |
| M2b single-cell loss < 0.05 but App-B′ regime stays `single` | eps-MSE is mis-aligned with the target geometry | Switch to direction-only loss or outcome-based (App-B′ regime as loss surrogate via REINFORCE-style score). |
| LoRA single-cell can't fit in 8000 steps | Supervision, not architecture | Re-examine cached Δ_t — consider direction-normalised target or seed-averaging. |

---

## 6. Open items (merged + de-duplicated)

| # | Question | Default | Why this matters |
|---|---|---|---|
| 1 | LLaVA install path | HF `llava-hf/llava-1.5-7b-hf`, lazy-load | If a local checkpoint exists, fewer GB downloaded. |
| 2 | VQAScore reduction | `min(p_yes)` over 3 questions | Conservative; chimera-discriminator. |
| 3 | App-C commit-column threshold | 0.5 (half locked-in) | 0.05 is too late to be informative for the grid. |
| 4 | GroundingDINO threshold | `box=0.35, text=0.25` | Lower to 0.20 only if App-B′ shows misses on the chimera seed. |
| 5 | M2 deployment metric of record | App-B′ regime + VQAScore min | Replaces CLIP-on-Tweedie. |
| 6 | Seed-averaging for Phase 3 | gated on cross-seed direction agreement >0.7 | Empirical, not a-priori. |

---

## 7. Cost summary

| Item | Compute | Wallclock |
|---|---|---|
| Thread A render (post-install) | GPU, ~6 min | 6 min |
| Thread A App-E sweeps (the only outstanding GPU work) | 1 GPU | ~2 h |
| GroundingDINO first download | ~700MB | 1–2 min |
| LLaVA-1.5 first download | ~14GB fp16 | 5–15 min depending on link |
| Thread B Phase 1 (M2b training, already kicked off) | 1 GPU | 1 day passive |
| Thread B Phase 2 (LoRA, conditional) | 1 GPU | 2–3 days |

Total for Phase 0 to ship a complete characterisation + a Phase-1 verdict on
learning: ~1 week.

---

## 7b. Thread C — Is Δ_t structured, or noise? (D-series diagnostics)

### The setup in plain words

You have a few seeds. For each seed, at every denoising step, you recorded
the "fix" — the thing you'd have to add to the broken PoE prediction to
turn it into the working mono-prompt prediction. Call that fix Δ_t. The
training cache has 50 of these per seed.

Two questions worth answering before training anything:

1. **Is Δ_t a structured object, or just noise?** If it's structured —
   smooth across time, low-rank, spatially focused — there's a real
   mechanism worth predicting. If it thrashes like white noise, "the fix"
   isn't really one thing.
2. **Are seeds asking for the same fix, or different ones?** If the same,
   a single shared model can be trained to predict it. If different,
   training is doomed to average incompatible targets.

The danger throughout is using a fancy method (PCA, fancy decompositions)
whose assumptions quietly change the question into a different one. The
plots below try to answer the actual questions as directly as possible,
and flag what each one can and can't tell you.

---

### Plot D1-A — does the fix point the same way at neighbouring timesteps?

**The idea.** If Δ_t is a real, slow-moving correction, the fix at step 10
and the fix at step 11 should point in nearly the same direction. If they
don't, "the fix at each step" is fifty unrelated objects that share a name.

**What it hopes to achieve.** Decide whether Δ_t is temporally coherent
before training a model that has to predict it smoothly over time.

**How the plot is laid out.** One curve. x-axis is denoising step t = 0…49.
y-axis is cos(Δ_t, Δ_{t+1}) ∈ [−1, +1]. Horizontal dashed line at 0 marks
the random-direction baseline.

**Patterns and what they mean.**

- Curve sits near +1 across the whole trajectory. Smooth, coherent fix. A
  time-conditioned model has an easy target.
- Curve dips toward 0 at specific steps. Those steps are where the fix
  changes character; train carefully around them.
- Curve bounces near 0 everywhere. The "fix at each step" is not one thing
  — it's per-step noise sharing a label.
- High in the basin-commit window (~steps 5–25), low elsewhere. The fix is
  coherent exactly when it matters.

**What the plot cannot tell you.** Whether the direction is the *right*
direction to flip the basin. It only tells you the fix is internally
consistent over time. Direction-correctness needs D4-A or App-E.

**Pre-committed pass:** mean cosine over steps 5–25 ≥ 0.85.

---

### Plot D1-B — does the trajectory of fixes live in a few dimensions or many?

**The idea.** Stack all 50 Δ_t vectors as rows of a matrix and ask: is this
essentially a few directions mixed in different proportions across time,
or is each step pointing somewhere genuinely new?

**What it hopes to achieve.** Decide whether the per-step fixes are a
tractable few-mode object (a small model head can in principle represent
them) or a high-dimensional mess (the model would have to memorise 50
unrelated targets).

**How the plot is laid out.** Bar chart. x-axis is "top-K components" for
K ∈ {1, 2, 3, 5, 10}. y-axis is cumulative SVD variance share captured by
those top-K components, on [0, 1]. Horizontal dashed line at 0.8.

**Patterns and what they mean.**

- Top-3 bar at ≥ 0.8. A handful of modes explain the whole trajectory.
  Tractable.
- Top-3 around 0.3, top-10 around 0.6. Long slow decay. Each step adds a
  genuinely new direction.
- Top-1 alone at ~0.9. The trajectory is essentially one direction with
  varying scale — simpler than expected.

**What the plot cannot tell you.** Whether the dominant directions are
*useful* directions. A trajectory can lie in a 3-dimensional flat that's
the wrong 3 dimensions. D1-C answers that.

**Pre-committed pass:** top-3 SVD variance share ≥ 0.8.

---

### Plot D1-C — does the fix line up with something we can compute without mono?

**The idea.** The fix is defined using mono, which we don't want to call
at deployment. So: is the fix approximately the same direction as some
quantity we already have without mono? Two natural candidates, both
already cached:

- `ε̃_J − ε̃_∅` — joint prompt minus unconditional. "What the joint adds
  over nothing."
- `ε̃_cat − ε̃_dog` — single-prompt difference. "Split the two concepts."

**What it hopes to achieve.** Determine whether the fix has a closed-form
approximation. If yes, the deployable method can be a one-line algebraic
combination of existing UNet calls — no training required.

**How the plot is laid out.** Two curves on shared axes. x-axis is
denoising step. y-axis is cosine similarity. Curve 1 is
cos(Δ_t, ε̃_J − ε̃_∅) per step. Curve 2 is cos(Δ_t, ε̃_cat − ε̃_dog) per
step. Horizontal dashed line at 0.5 marks the "well-aligned" threshold.

**Patterns and what they mean.**

- One curve stays ≥ 0.5 across the basin-commit window. The fix is
  approximately a scaled version of that vector. Mechanism-level result;
  closed-form deployment is on the table.
- Both curves near zero. The fix is not a simple combination of existing
  UNet outputs. Learning is required.
- A curve is high outside the commit window and low inside it. Misleading
  agreement — the alignment is where the fix doesn't matter.

**What the plot cannot tell you.** How good a closed-form approximation
would actually be at flipping the basin. High cosine and "actually works
at inference" are different questions; the latter takes a substitution
run like D4-A.

**Pre-committed pass:** at least one candidate stays cos ≥ 0.5 across the
basin-commit window.

---

### Plot D2 — where in the image does the fix live?

**The idea.** The fix is a tensor with spatial dimensions. If PoE fails by
mashing the cat and dog into the same place, the fix should be concentrated
exactly where that collision happens. If it's spread uniformly, the fix is
doing something else (or nothing).

**What it hopes to achieve.** Decide whether the fix has a spatial
mechanism (acts at the cat–dog collision) or a global mechanism
(everywhere, equally). The first gives a clean story; the second is less
interpretable.

**How the plot is laid out.** Four heatmaps side by side, one per chosen
timestep (t ∈ {5, 15, 25, 35}). Each heatmap is 128×128 pixels (latent
spatial resolution); pixel intensity is the L2 norm across the 4 latent
channels of Δ_t at that pixel. Two coloured rectangles overlay each
heatmap — the GroundingDINO boxes for "cat" and "dog" detected in the
decoded x̂_0 at that step.

**Patterns and what they mean.**

- Bright pixels concentrate inside the cat–dog box overlap. The fix
  targets exactly the collision region.
- Bright pixels uniform across the image. The fix is a global nudge; no
  spatial story.
- Bright pixels in the background corners, far from either box. The fix
  is doing something irrelevant (or the boxes are wrong).
- The bright region drifts from "everywhere" at early t to "the collision"
  by mid t. The fix becomes spatial as the basin commits.

**What the plot cannot tell you.** What the fix is doing inside the
collision region — flipping cat to dog, sharpening edges, separating,
something else. L2 norm hides direction; only the magnitude is here, not
what it's pushing toward.

---

### Plot D3 — do different seeds want the same fix, or different fixes?

**The idea.** For each pair of seeds, compute the cosine similarity of
their Δ_t vectors at each step. Render as a heatmap — quick first pass at
"is the fix a property of the prompt pair, or of one trajectory."

**What it hopes to achieve.** First-pass answer to whether seeds agree at
all. Necessary but not sufficient for the Phase-3 seed-averaging decision
in §3; D4-A is the sufficient check.

**How the plot is laid out.** Heatmap. Rows are the three seed-pairs
(4–vs–42, 4–vs–123, 42–vs–123). Columns are denoising steps 0–49. Cell
colour is cosine similarity on [−1, +1] (diverging colormap centred at 0).

**Patterns and what they mean.**

- Mostly warm cells (cos ≥ 0.5) across all pairs and most steps. Seeds
  agree on direction; the fix is a property of the prompt pair.
- Mostly near zero. Each seed's correction is incomparable to the others.
- Warm in the basin-commit window, cool elsewhere. Seeds agree exactly
  where it matters — the cleanest possible outcome.
- One row consistently cooler than the other two. One seed is the odd one
  out; cross-reference its end-result failure mode.

**What the plot cannot tell you.** Whether the level of agreement is
*enough* to train a shared model. Cosine 0.6 sounds high in absolute
terms but says nothing about whether substituting one seed's fix for
another's actually flips the basin. D4-A answers that.

**Pre-committed pass:** cooperative-pair (butterfly × flower_meadow)
cross-seed cosine averages ≥ 0.5 over basin-commit. (Cat × dog is allowed
to fail this for the legitimate reason in addition C1.)

---

### Plot D4-A — the taste test: does the shared fix actually work?

**The idea.** Instead of measuring whether seeds' fixes "look similar"
under some decomposition, just *use* the average fix and see if it flips
the basin. This is the only D-plot that answers the question in the same
coordinates as the downstream training objective.

**What it hopes to achieve.** Decide cheaply, before training anything,
whether the residual-prediction approach has a stable target. If the
leave-one-out mean fix works almost as well as a seed's own fix, the
simplest possible shared model is on the table. If not, the whole
seed-averaging plan in §3 is doomed.

**How the plot is laid out.** Bar chart. Three seed groups along the
x-axis (seeds 4, 42, 123). Within each group, four bars side by side, one
per condition:

- **Oracle:** the seed's own cached fix — ceiling.
- **Shared-mean:** average of the *other* two seeds' fixes — what a
  trivial shared model would predict.
- **Zero:** no fix — PoE baseline floor.
- **Shuffle:** another single seed's fix used in full — idiosyncratic
  sanity check.

Primary y-axis is VQAScore min (§4 protocol) on [0, 1]. **Per-seed**
horizontal threshold line at zero_seed + 0.7·(oracle_seed − zero_seed) —
seed-relative, not global, since each seed has its own PoE floor and
oracle ceiling, and the pre-committed pass criterion is defined relative
to those two anchors. Above each bar, the corresponding VQAScore *mean*
is printed in small grey text alongside min, as a sanity check that
"min" isn't being moved by a single failed prompt. Below the bars, a
single-row strip with one filled/hollow disc per bar encoding the
detection regime (filled = `both_distinct`, hollow = anything else;
classifier from App-B′).

**Patterns and what they mean.**

- `oracle ≳ shared-mean ≫ shuffle ≈ zero`, shared-mean bar above the
  per-seed threshold line, filled discs under oracle and shared-mean.
  The shared fix recovers the basin flip without ever seeing the
  held-out seed's data. Seed-averaging is licensed.
- `shared-mean ≈ shuffle`, both short. Seeds want individual corrections.
  Phase-3 seed-averaging cannot work even if D3 cosine looked moderate.
- All four bars the same height regardless of value. The cell is
  degenerate for separating conditions. Pick another headline cell.
- `shared-mean > oracle`. Averaging across seeds regularises away a
  per-seed quirk. Rare; most likely a caching / indexing bug — diagnose
  before celebrating.
- Filled disc under shared-mean but VQAScore-min bar low. Detection
  classifier and VQAScore disagree on this seed. Trust the metric tied
  to the downstream training objective (VQAScore); flag the discrepancy
  rather than picking the friendlier number.

**What the plot cannot tell you.**

- Whether the gap between shared-mean and oracle is closed-formly fixable
  (a per-seed scalar) or fundamentally per-seed (different direction).
  D4-B disentangles that.
- N = 3 is fragile. Shared-mean is a mean of two; shuffle draws from the
  same tiny pool. The four conditions are less independent than the
  panel structure implies. The "≥ 2/3 seeds" pre-committed criterion is
  the entire statistical defence; a marginal 2/3 result is suggestive,
  not decided.
- Near-ceiling compression. If oracle and shared-mean both land near
  0.95, the visual gap looks small but the failure-rate ratio
  (1 − VQAScore-min) may have doubled. When bars cluster ≥ 0.9, add an
  inset on 1 − VQAScore-min (log axis) or print failure-rate ratios
  above the bars.
- Single-scalar collapse hides *where* shared-mean wins. A high bar can
  come from "works on every prompt" or "works on the easy prompts and
  PoE handles the rest." If §3 training-data construction depends on
  the distinction, add a per-prompt scatter as a supplementary panel.
- Shuffle is an asymmetric pairing with N = 3. State the chosen pairing
  explicitly in the caption (e.g., "shuffle for seed 42 = seed 4's
  fix"); ideally show the verdict is invariant under pairing choice.

**Pre-committed pass:** shared-mean reaches ≥ 0.7 × (oracle − zero) on
VQAScore min on ≥ 2 of 3 seeds, AND shared-mean produces `both_distinct`
on ≥ 2 of 3 seeds.

---

### Plot D4-A-t — when in the trajectory does the shared fix have to be in place?

**The idea.** D4-A swaps the fix at every step and reads off a single
scalar at the end. If shared-mean works there, the followup question is:
*was the work being done by the swap during the basin-commit window, or
distributed across the whole trajectory?* This plot localises D4-A's
verdict in time by repeating the substitution only inside a chosen window
and leaving PoE untouched outside it.

**What it hopes to achieve.** Tell us, if shared-mean is going to be
deployed, *which steps* it actually has to be accurate at. If only the
commit window matters, the deployed predictor only needs to be accurate
there. If all steps matter equally, accuracy must be uniform across t —
a much harder training target.

**How the plot is laid out.** Same four conditions as D4-A (oracle,
shared-mean, zero, shuffle), but the per-step swap is restricted to a
window W. Outside W, every condition reverts to plain PoE — so the floor
is always "PoE-elsewhere," and the question becomes "what does activating
condition C inside W buy you over PoE-everywhere?"

Repeat for four windows, anchored to the v3 commit timing:

- **pre-commit:** t ∈ [0, 5]
- **commit:** t ∈ [5, 25] (the basin-commit window from Fig 2 / App-C)
- **post-commit:** t ∈ [25, 49]
- **all:** t ∈ [0, 49] (= the original D4-A; included as the reference column)

**Four panels in a row** (small multiples), one per window, with `all`
as the rightmost reference panel. Inside each panel: three seed groups
(4, 42, 123) along the x-axis, each with four bars in D4-A's fixed
colour scheme (oracle / shared-mean / shuffle / zero). y-axis is
VQAScore min on a **shared [0, 1] range across all four panels** —
critical for column-height comparability by eye; without this, the
"shared-mean rises off the floor only inside commit" signal is invisible.

**Per-panel** horizontal threshold line at zero_panel + 0.7·(oracle_panel
− zero_panel), drawn per-panel rather than per-seed because the question
inside a window is whether shared-mean tracks oracle *within that
window*. (D4-A's threshold is per-seed; D4-A-t's is per-panel. Different
question.)

Above each panel, an annotation `t ∈ [a, b] · N_steps = b − a` so the
reader knows window width without reading the caption. Below each
panel, the same filled/hollow detection-regime strip as D4-A. A thin
grey "plain PoE everywhere" reference bar drawn in each panel anchors
the zero bar visually to the do-nothing floor and makes inter-panel
zero variation (which should be near-zero if wiring is correct) visible.

**Patterns and what they mean.**

- shared-mean tracks oracle in the **commit window**, both well above zero;
  shared-mean ≈ zero in pre-/post-commit. The shared fix's value is
  concentrated where Fig 2 said it would be. Deploy a predictor accurate
  on steps 5–25 and don't worry about the rest.
- shared-mean tracks oracle in **all four** windows. The fix is
  distributed; predictor accuracy must be uniform across t. Harder
  training target, but at least the target is consistent.
- shared-mean tracks oracle in **pre-commit but not commit**. Surprising —
  the work happens before basin commit. Re-examine Fig 2's commit timing
  before trusting this; it's likelier a wiring bug (off-by-one in step
  indexing, or override applied to the wrong tensor).
- shared-mean ≈ zero in every window but oracle ≫ zero in commit. The
  oracle works because of per-seed specifics that the cross-seed mean
  smooths away. Seed-averaging is dead; consider per-seed conditioning or
  a stronger pooled target (e.g. median, robust mean).
- zero already reaches `both_distinct` inside some window. The cell is
  sensitive to *anything* during that window, and the conditions aren't
  separable there. Pick a stricter cell or narrow the window.
- The `all` panel disagrees with D4-A's bars. Treat as a wiring failure
  — D4-A-t's `all` column must reproduce D4-A exactly. Diagnose before
  trusting any other panel.

**What the plot cannot tell you.**

- Sub-window resolution. "Commit [5, 25]" is 20 steps wide; whether the
  work is at step 7 or step 22 is hidden. If a per-step map is needed,
  run a sliding-window variant of the same code (more GPU, no new logic).
- Cross-window interaction. A hybrid "shared-mean in commit, oracle in
  post-commit" is plausibly better than either alone but isn't on the
  figure. Separate ablation if the windows separate cleanly.
- Validity of the window edges. The four windows inherit Fig 2's
  basin-commit timing. If Fig 2 is wrong (commit really at [3, 18]),
  the partitioning will smear signal across panels and the commit panel
  may look unimpressive. Cross-reference Fig 2 in the caption.

**Method discipline (why this is principled).** The substitution-test
framing from D4-A is preserved exactly — same conditions, same metric,
same coordinates as the downstream training objective. The only change is
*which steps* the swap is active for, which is a localisation, not a new
metric or a decomposition. Nothing here assumes linearity, low-rank
structure, or that "the fix" decomposes additively across steps; if it
doesn't, the windowed scores will just be lower than the all-window
scores, which is itself an interpretable outcome.

**Critical wiring requirement.** All 16 runs per seed (4 windows × 4
conditions) **must share the same noise schedule and the same initial
latent** for that seed. Otherwise inter-panel `zero` variation will
produce visible artifacts that get mistaken for window effects. The
current [`scripts/run_veracity_phase_c.sh`](scripts/run_veracity_phase_c.sh)
pins seeds; the D4-A-t orchestrator must additionally pin the initial
latent and DDIM step sequence across (window, condition) within each
seed. If the `zero` reference bar visibly differs across panels, the
wiring is wrong, not the result.

**Pre-committed pass:** commit-window shared-mean reaches ≥ 0.7 ×
(commit-window oracle − commit-window zero) on VQAScore min on ≥ 2 of 3
seeds. (D4-A's all-window threshold remains the headline; D4-A-t's
commit-window threshold licenses "the predictor only needs to be accurate
in the commit window" as a *deployment* simplification.)

**Cost note.** Four windows × four conditions × N seeds = 16 × N sampler
runs. With the same per-step `delta_override` plumbing as D4-A, this is
strictly more GPU but no new code. Budget ~6 × 4 = 24 GPU-hours for the
headline cell if the four windows are run independently; ~half that if
the inner loop is structured to reuse the noise schedule across windows.

---

### Plot D4-B — do seeds disagree on direction, or on size?

**The idea.** Two seeds can "disagree" in two very different ways: same
direction but different lengths (fixable with a scalar), or different
directions (fatal for a shared model). This figure splits them apart.

**What it hopes to achieve.** When D4-A is ambiguous, decide whether the
residue is a "rescale per seed" problem (a single per-seed gain term
solves it) or a "different fix per seed" problem (a single shared model
cannot).

**How the plot is laid out.** Two panels, stacked, sharing the t-axis.

- **Top panel:** one faint line per seed, plotting ‖Δ_t‖ vs t. Median ±
  IQR band overlaid. Tells you where in t the fix is large.
- **Bottom panel:** one line per seed, plotting cos(Δ_t^{(s)}, Δ̄_t^{loo})
  vs t — alignment of each seed with the leave-one-out mean. A grey null
  band sits behind: the cosine you'd get by chance, computed by shuffling
  timesteps before correlating (controls for the high-dimensional ambient
  space).

**Patterns and what they mean.**

- Top large, bottom well above the null, both in the same t-window. Seeds
  agree on direction precisely where the fix matters. Disagreement is in
  magnitude only — fixable with a scalar.
- Top large, bottom inside the null band. Seeds disagree on direction at
  exactly the steps that matter. Fatal for a shared model.
- Top small everywhere. The fix is tiny on this cell; everything else is
  over-interpretation.
- Bottom high outside the basin-commit window, low inside it. Misleading
  aggregate cosine — the agreement is where the fix doesn't matter.

**What the plot cannot tell you.** Whether the magnitude variation can be
predicted from anything observable at inference. If seeds need different
scalars and you don't know which seed you're in until after sampling,
"fixable with a scalar" is theoretical only.

**Pre-committed pass:** bottom-panel mean cos(seed, loo-mean) ≥ 0.5 over
steps 5–25 on the cooperative pair. (Cat × dog allowed to fail for the
same reason as D3.)

---

### Plot D4-C — are there two camps of seeds wanting different fixes?

**The idea.** Averages hide multimodal structure. If seeds split into two
groups — half wanting fix-A, half wanting fix-B — a mean across all of
them looks like nothing in particular, and bulk cosine looks "moderate."
A pairwise heatmap with consistent ordering exposes the cluster structure.

**What it hopes to achieve.** Detect whether the fix is one thing, or a
small number of things gated by something we haven't conditioned on. If
two clusters appear, the right architecture is a mixture or routing head,
not a single shared model.

**How the plot is laid out.** Five small heatmaps in a row, one per chosen
timestep (t ∈ {5, 15, 25, 35, 45}). Each heatmap is N×N (seeds × seeds);
cell colour is pairwise cosine similarity. Critically, row/column order is
set by hierarchical clustering of *one* panel and reused across all
panels, so seeds appear in the same position throughout and cluster
membership is visually trackable.

**Patterns and what they mean.**

- All five panels uniformly warm. One shared fix.
- Persistent two-block structure (warm top-left and bottom-right blocks,
  cool off-diagonals). Two camps. Need a mixture model or a per-camp head.
- Cluster structure forms in mid t and dissolves by late t. The fix
  branches during commit and merges afterward — consistent with "the
  failure mode is decided in the commit window."
- Cluster structure persists at all t but seeds rotate between camps. The
  clustering you're seeing is noise; redo with more seeds.

**What the plot cannot tell you.** *Why* seeds cluster. The grouping might
track end-result failure mode (pure-cat vs pure-dog vs chimera), but the
heatmap doesn't show that — you have to cross-reference against the
per-seed final image to know.

---

### Plot D4-D — PCA, only if needed, only with guards (skip by default)

**The idea.** PCA is the standard tool for decomposing variation, but on
this specific question it has three traps that can quietly answer the
wrong question:

1. **Centring removes the very signal we want.** PCA usually subtracts the
   mean first; the mean is exactly what a shared model would predict.
2. **Variance ratios are inflated** when there are very few seeds (N=3)
   and very high dimensions (d ~ 65k) — rank is bounded above by N−1, so
   "top-3 explains 90%" is mechanical, not informative.
3. **Low-rank ≠ agreement.** Seeds can live in a 3-dimensional flat that
   contains three very different directions.

If PCA is run anyway, it must be run with explicit guards.

**What it hopes to achieve.** When D4-A and D4-B are inconclusive, give a
careful second look at the linear structure. In the default workflow this
plot is omitted — D4-A is more direct and D4-B more interpretable.

**How the plot is laid out.** Three curves over t (steps 0–49):

- Curve 1: uncentred top-1 explained variance ratio, with a permutation-
  null band drawn behind it (the null comes from shuffling entries within
  each Δ_t and recomputing).
- Curve 2: angle between uncentred PC1 and Δ̄_t (the cross-seed mean
  direction). Smaller is better.
- Curve 3: angle between centred PC1 and uncentred PC1.

**Patterns and what they mean.**

- Curve 1 well above the null band; Curve 2 near zero. The dominant
  linear direction *is* the shared mean — good for a shared model.
- Curve 1 above the null band but Curve 2 far from zero. There's a strong
  direction, but it's not the mean — a shared mean would miss it.
- Curve 1 inside the null band everywhere. No linear structure beyond
  random shuffling. The shared-model story has no linear-algebraic
  backing.
- Curve 3 large. The mean is the whole story; centred PCA is uninformative
  here, and any conclusion that uses centred PCA is studying noise.

**What the plot cannot tell you.** Whether non-linear structure (mixtures,
low-rank-plus-sparse, manifolds) is present. PCA only sees the first-order
linear story. If D4-C showed cluster structure but D4-D looks "fine,"
trust D4-C.

### Methodological additions

**(C1) Run D3 on a cooperative pair, not just cat × dog.** The three cat × dog
seeds fail in three qualitatively different ways (pure-cat, pure-dog,
chimera). Even a clean pair-coherent Δ_t would give different corrections
because it would be correcting three different failures — cross-seed cosine on
cat × dog alone is over-determined to look low. Fix: also compute D3 on
butterfly × flower_meadow (the cooperative pair from existing controls). If
cross-seed cosine is high on cooperative and low on collision, Δ_t is
pair-coherent and seeds only diverge when the pair induces divergent
failures — a much more interesting result than "Δ_t is seed-noisy." If both
pairs are low, the correction is genuinely seed-specific.

**(C2) Pre-commit pass/fail thresholds before plotting:**

- D1-A passes iff mean cosine over steps 5–25 ≥ 0.85.
- D1-B passes iff top-3 SVD variance share ≥ 0.8.
- D1-C passes iff at least one Mono-free candidate stays cos ≥ 0.5 across the
  basin-commit window.
- D3 passes iff cooperative-pair cross-seed cosine averages ≥ 0.5 over
  basin-commit.

Pre-committing forces decisions instead of post-hoc squinting.

**(C3) No budget conflict with Fig 4 / App-B′ / App-E.** Diagnostics are
CPU-only and reuse what's already on disk: residuals computable from cached
`eps_{a,b,j,uncond}_raw`, direction-stability idiom at
[poe_repair/experiments/veracity/metrics.py:190](poe_repair/experiments/veracity/metrics.py#L190),
spatial-heatmap idiom at
[poe_repair/experiments/idea5b/figures.py:189](poe_repair/experiments/idea5b/figures.py#L189),
training-cache layout at
[outputs/training_cache/heldout/a_cat__x__a_dog/seed_42/residuals/](outputs/training_cache/heldout/a_cat__x__a_dog/seed_42/residuals/)
(50 steps cached per seed). Build is ~1 pomodoro of code, minutes of runtime.

### What is done vs missing (audit, 2026-05-11)

Data substrate:

- ✓ Cat × dog raw eps caches: `outputs/training_cache/heldout/a_cat__x__a_dog/seed_{4,42,123,...}/residuals/step_{000..049}.pt`, keys `{x_t, eps_a_raw, eps_b_raw, eps_j_raw, eps_uncond}` — verified present.
- ✗ **Cooperative-pair (butterfly × flower_meadow) raw eps cache is MISSING from `outputs/training_cache/`.** Butterfly × meadow exists only under `outputs/veracity/` in the `teacher_residual_const_lamXXX` format, which is the wrong layout for these diagnostics. Required for addition C1 / D3-cooperative.
- ✗ Δ_t tensors themselves are not pre-materialised; the cache stores raw eps and Δ_t must be computed on-the-fly as `guided_eps(eps_j_raw, eps_uncond) − ε̃_PoE(eps_a_raw, eps_b_raw, eps_uncond)`. Cheap, but no helper exists.

Code (the old diagnostics module — SVD energy, in-plane fraction, low-pass — was deliberately pruned per [poe_repair/diagnostics/residual.py:1-9](poe_repair/diagnostics/residual.py#L1-L9); only `attention_overlap` remains):

- ◐ D1-A consecutive-step cosine: [`direction_stability_matrix`](poe_repair/experiments/veracity/metrics.py#L190) returns the full T×T cosine matrix but operates on `payload["delta"]` from the *veracity* run layout, not the training-cache layout. Needs a small adapter that computes Δ_t from raw eps and extracts the super-diagonal. **Plot not built.**
- ✗ D1-B SVD energy bar chart: not implemented (deleted with the pruned diagnostics module).
- ✗ D1-C basis-alignment curves (Δ_t vs ε̃_J−ε̃_∅, Δ_t vs ε̃_cat−ε̃_dog): not implemented.
- ◐ D2 spatial heatmap: [`fig06_spatial_heatmap`](poe_repair/experiments/idea5b/figures.py#L189) does the L2-over-channels idiom but on the CLIP-guidance *force*, not Δ_t, and **does not overlay GroundingDINO boxes**. Reusable shell; new wiring needed.
- ✗ D3 cross-seed cosine heatmap: not implemented.
- ✗ Pre-committed thresholds (C2): not yet encoded as gates anywhere — would naturally live next to the decision matrices in §3 / `diag0_overfit_replay`.

GroundingDINO + decoded x̂_0 for D2 box overlay: helpers exist (`detect_boxes`, `classify_detection_regime` in [veracity/metrics.py](poe_repair/experiments/veracity/metrics.py)) but have never been called against a *decoded training-cache* x̂_0; only against full-run final images. Trivial bridge.

D4 status:

- ✗ D4-A shared-mean substitution: requires a per-step `delta_override` argument plumbed through the inner sampling loop. Veracity's λ-walk in App-E injects a *scaled* Δ over a window, not a per-step swap with a foreign tensor, so the existing injection code is not directly reusable. New code path. ~½ day build + ~30 min GPU per cell × 3 seeds × 4 conditions = ~6 GPU-hours for the headline cell.
- ✗ D4-A-t windowed substitution: same code path as D4-A, just gates the override by `t ∈ window`. Adds 3 extra windows × 4 conditions × 3 seeds = 36 extra sampler runs ≈ ~18 additional GPU-hours (or ~½ that if the noise schedule is reused across windows in the inner loop). No new code beyond a window argument.
- ✗ D4-B per-seed-vs-loo-mean direction split: pure post-hoc on cached eps, ~30 min code once the Δ_t loader (Thread C item 2) exists.
- ✗ D4-C cluster-ordered cosine grid: pure post-hoc on cached eps, ~30 min code, same dependency.
- ✗ D4-D PCA with guards: skip by default; only build if D4-A and D4-B are inconclusive.

**Net outstanding work to ship Thread C (D1–D4):**

1. Build butterfly × flower_meadow into `outputs/training_cache/` for seeds {4, 42, 123} via [`scripts/build_training_cache.py`](scripts/build_training_cache.py). Single GPU pass, ~hour.
2. Write a thin Δ_t loader on top of the training-cache layout (raw eps → guided ε̃_J, ε̃_PoE → Δ_t per step).
3. Implement D1-A / D1-B / D1-C / D2 / D3 / D4-B / D4-C as a single CPU-only module (best home: a new `poe_repair/experiments/thread_c_structure/` mirroring the veracity layout, or extending [veracity/metrics.py](poe_repair/experiments/veracity/metrics.py) directly).
4. Wire a per-step `delta_override(t)` argument through the inner sampling loop (gated by an optional `window=(t_lo, t_hi)`) and a thin orchestrator for D4-A's four conditions × N_seeds (window = all) and D4-A-t's four conditions × 3 extra windows × N_seeds. Same JSON consumed by §4. ~1 pomodoro of code + ~6 GPU-hours for D4-A + ~18 GPU-hours for D4-A-t (or ~½ that with noise-schedule reuse).
5. Encode the C2 thresholds and the D4-A / D4-A-t / D4-B thresholds as a verdict JSON written alongside the figures so Phase-1 / Phase-3 decisions in §3 can read them mechanically.

Items 2, 3, 5 are the ~1-pomodoro CPU build. Item 1 is GPU and gates the
cooperative-pair half of D3 / D4-B. Item 4 is the only D4 GPU spend; D4-A
gates the Phase-3 seed-averaging decision in §3, and D4-A-t gates the
"predictor only needs to be accurate in the commit window" deployment
simplification.

---

## 7c — Cross-seed feasibility figures (PCA grid and VLM-projection grid)

Two paired figures that together decide which loss family is feasible for the
shared residual-prediction model. The PCA grid screens for L2-on-Δ_t
feasibility; the VLM-projection grid screens for outcome-supervised
(DRaFT/AlignProp/DDPO) feasibility. They answer different questions on the
same cached data and are most informative when they disagree.

### Method 1: The PCA grid

**The idea.** You have collected, for each seed, a residual vector Δ_t at
every timestep — the correction that turns the broken PoE prediction into the
working mono-prompt prediction. The question driving this figure is: are the
seeds asking for the same correction, or different ones? If they're all
asking for roughly the same thing, a single shared model can be trained to
predict that correction. If they're asking for wildly different things, a
shared model has no consistent target and squared-error training will fail.

**What it hopes to achieve.** Decide, cheaply and before training anything,
whether the residual-prediction approach has a stable target. It's a
feasibility check for the simplest possible training plan: regress on Δ_t
with squared-error loss.

**How the plot is laid out.** A 6-panel grid, one panel per timestep across
the diffusion trajectory (e.g., t = 900, 700, 500, 300, 150, 50). Inside each
panel, each seed contributes one dot, positioned by where its Δ_t at that
timestep lands in the top two principal-component directions computed from
the across-seed residual matrix at that timestep. Axes are "PC1" and "PC2,"
meaningful only within the panel. Color encodes seed identity so individual
seeds can be tracked across panels.

**Patterns and what they mean.**

- Tight cluster within a panel. Seeds agree numerically. Squared-error
  training has a stable target at this timestep.
- Diffuse scatter within a panel. Seeds disagree numerically. Squared-error
  training will fit the average, which is unlikely to be any individual
  seed's correct answer.
- Tight in some panels, scattered in others. The residual is well-defined in
  some parts of the trajectory and ill-defined in others. Train only on the
  tight-cluster timesteps.
- Tight clusters that drift smoothly across panels. The residual changes
  shape over the trajectory but does so consistently — good sign for a
  time-conditioned model.
- Tight in pre-commit panels, scattered inside the commit window. Backwards
  from what you want. The commit moment is exactly where you need a sharp
  consistent correction; scatter there is a red flag.

**What the plot cannot tell you.** Whether the agreement you see is on the
part of the residual that matters. PCA picks directions of maximum numerical
variance, which in latent ε space is dominated by high-frequency texture
noise the decoder will partly erase. The "tight cluster" might be tight on
noise the model doesn't need to learn, while the semantically important
low-frequency component sits in a smaller-variance direction PCA hides. The
plot answers "are the vectors close in raw numerical sense?" but not "are
they close in the sense that matters for the final image?"

### Method 2: The VLM-projection grid

**The idea.** Stop measuring whether the residuals look alike as tensors and
start measuring whether they do the same thing to the picture. For each seed
and each injection strength α, generate the final image that residual
produces, hand it to the same kind of grader that decides
chimera-vs-co-occurrence, and let the grader's scores be the plot's
coordinates. "Similar" now means "similar in the eye of the discriminator
that actually decides the task" — the only definition of similar that
matters.

**What it hopes to achieve.** Decide whether the *effect* of the residual is
consistent across seeds, even when the residual tensor isn't. This is a
**necessary condition, not a sufficient one**, for a more forgiving training
plan: instead of L2-matching Δ_t, train the student to produce *any* residual
that achieves the same outcome and reward it on outcome quality (DRaFT,
AlignProp, DDPO-style). That plan is only worth attempting if the outcome
itself is consistent across seeds. A green light authorizes the experiment;
it does not predict it will succeed — the student conditions on (z_t,
prompt), while the oracle conditions on the mono-trajectory, and the oracle
outcome being shared does not imply the student has the inputs to reproduce
it.

**Protocol (locked).** Three things have to be pinned down or the plot has no
defined semantics:

1. **Injection window.** For panel t, inject α · Δ_t only at timestep t. All
   other timesteps run vanilla PoE. Panel index = injection time, not decode
   time. Without this, the plot conflates per-timestep effect with cumulative
   effect.
2. **Completion sampling.** After the injected step, finish the trajectory
   with DDIM, η = 0 (deterministic). No DDPM stochasticity in the tail —
   otherwise arrow length conflates residual effect with sampling noise.
3. **Decode.** Decode the final z_0. Never decode an intermediate z_t and
   never use Tweedie x̂_0(z_t) for grading — detectors collapse on blurry
   x0-predictions at high t and the y-axis flatlines.

**Grader (calibrated).** Two scores per image, each chosen to be smooth in α:

- **x = co-occurrence score** — VQAScore-style probabilistic grader (yes/no
  token logits on "Does this image show one [A] and one [B] as separate
  animals?"). Avoid raw CLIP cosine — known to saturate and behave like a
  switch.
- **y = separation confidence** — max P(box_A) · max P(box_B) from an
  open-vocabulary detector (OWL-ViT or GroundingDINO).

**Calibration as a blocking gate.** Before the main grid, on 2–3 seeds at the
strongest commit-window timestep, sweep α ∈ {0, 0.25, 0.5, 0.75, 1} and plot
each axis against α. If either is non-monotone, swap the grader (try
embedding distance to "two-animal" exemplars) until monotone. A non-monotone
grader cannot rank seeds and the plot is uninterpretable. **Recalibrate at
any other timestep where the figure looks suspicious** — a grader that's
monotone at t=400 is not guaranteed monotone at t=150 or t=700, and a flat
axis at high t might be the grader, not the residual.

**Plot layout.** Same 6-panel grid skeleton, one panel per injected timestep.
Inside each panel:

- X-axis: co-occurrence score. Low = chimera, high = clean co-occurrence.
- Y-axis: separation confidence. Low = single blob, high = two well-separated
  objects.

For each seed, three points (α = 0, α_partial, α = 1) connected by an arrow
baseline → partial → oracle. Expected geography: baselines pile up
bottom-left, oracles pile up top-right, partial points sit on the path
between.

**Two enrichments that turn this from qualitative to quantitative.**

- **Confidence ellipses.** Five reruns per (seed, t, α), each with a freshly
  reseeded z_T (DDIM η=0 is fixed; the variation is in the starting noise).
  Plot 95% ellipses around each point. Arrow length is only meaningful
  relative to ellipse size — **an arrow shorter than its own ellipses is not
  a finding.** Three reruns is the bare minimum to fit an ellipse to; use
  five when the finding will load-bear.
- **Route tag on the arrowhead.** For the α = 1 run, log which
  cross-attention map (token A vs. token B) gained the most mass relative to
  baseline. Color the arrowhead accordingly. Two arrows landing top-right
  with opposite tags are qualitatively different from two with the same tag —
  this is the cheapest patch for the 2D-collapse problem. **For seeds where
  both tokens gained roughly equally (high route-tag entropy), draw the
  arrowhead hollow or grey** so they're visually distinguishable from
  clean-tagged seeds and don't silently pollute the "same route" reading.

**Seed filter.** Drop any seed whose baseline (x₀, y₀) is already near the
top-right threshold — no chimera to fix, no signal to extract.

**Reading the geography (commit-window panels).**

| Pattern | Meaning | Decision |
|---|---|---|
| Arrows parallel, similar length, small ellipses, same route tag | Shared direction, shared magnitude, shared route | Green light for outcome-supervised training |
| Arrows parallel, lengths vary, same route tag | Shared direction, seed-specific magnitude | Green-ish — learned magnitude head or per-seed normalization |
| Arrows parallel, opposite route tags | Same coordinates, different internal fix | Yellow — shared model picks one route; expect coverage loss |
| Arrows fan in different directions | Seeds want different outcomes | Red light |
| Arrows long but inconsistent direction inside commit window | Powerful but seed-dependent intervention | Red light — outcome supervision will likely fail |
| Arrow length < ellipse radius | Measurement noise > effect | Not a finding; either α_partial is wrong or grader is saturated |
| Many hollow/grey arrowheads | Fixes route through background/layout, not subject tokens | Route-tag is uninformative here; rely on coordinates and ellipses only |

Pre-commit panels are expected to show short arrows regardless. Long,
consistent arrows arriving only inside the commit window is second-order
confirmation of the commit-window picture, on semantic axes.

**What the plot still cannot tell you.**

- **Grader is monotone, not linear.** Calibration enforces monotonicity, not
  linearity. Arrow length is ordinal across seeds within a panel, not a
  quantitative effect size.
- **Necessary, not sufficient.** Maximally consistent arrows prove the oracle
  outcome is shared. They do not prove a student conditioned only on (z_t,
  prompt) has enough information to produce that outcome. The oracle
  residual may depend on features of the mono-trajectory the student never
  sees.
- **Route ambiguity beyond two tokens.** The cross-attention tag is a 2-class
  summary. If a residual works by changing background or shifting layout,
  the tag is misleading — flagged by the hollow-arrowhead convention but not
  resolved.

**What "success" unlocks.** If the commit-window panels show parallel,
similar-length, same-route arrows with small ellipses, the next experiment is
outcome-supervised fine-tuning of the residual head via a DRaFT/AlignProp-
style differentiable reward through the VLM grader (or DDPO if
non-differentiable). If the panels show any red-light pattern, abandon the
outcome-supervised plan and stay with L2-residual supervision, accepting its
per-seed coverage limit.

### How methods 1 and 2 relate

The two figures answer two different feasibility questions about two
different training plans:

| Figure | Question it answers | Training plan it screens for |
|---|---|---|
| PCA grid (latent space) | Are the residual *tensors* consistent across seeds? | Squared-error regression on Δ_t |
| VLM-projection grid (semantic space) | Are the residual *effects* consistent across seeds? | Outcome-supervised (DRaFT / AlignProp / DDPO) |

The interesting outcomes are the disagreements. **PCA scattered, VLM
aligned** → seeds disagree numerically but want the same outcome;
squared-error will fail, outcome-supervised has a real target. **PCA
aligned, VLM scattered** → residuals look consistent but produce inconsistent
images; the injection scheme itself is unstable rather than the loss being
wrong. Agreement on both is the rosy case; disagreement on both means the
residual is noisy and the outcomes are also noisy, and there isn't much to
train.

Run the PCA grid first — it's CPU-only on cached eps and shares the Δ_t
loader from §7b. Run the VLM-projection grid after LLaVA / GroundingDINO are
installed (already a §4 dependency), and use it as the primary cross-seed
consistency figure for the consolidated plan, with the PCA grid relegated to
an appendix as a latent-space check.

---

## 8. Quick links

- v3 figure spec (deep detail): [veracity-figure-plan.md](veracity-figure-plan.md)
- Original M2 phased plan (decision matrices): [m2-residual-diagnostic-plan.md](m2-residual-diagnostic-plan.md)
- Figure code: [poe_repair/experiments/veracity/figures.py](poe_repair/experiments/veracity/figures.py)
- Eval helpers (detection + VQA): [poe_repair/experiments/veracity/metrics.py](poe_repair/experiments/veracity/metrics.py)
- App-E sweep script: [scripts/run_app_e_window_injection.sh](scripts/run_app_e_window_injection.sh)
- Diagnostic 0 orchestrator: [poe_repair/experiments/diag0_overfit_replay/main.py](poe_repair/experiments/diag0_overfit_replay/main.py)
- Active execution plan (separate): [.claude/plans/serialized-seeking-pebble.md](.claude/plans/serialized-seeking-pebble.md)
