# Layer 12, step 7: the maths drip

⬅️ [Layer 12, step 7](../12-the-correction-at-scale.md#7-you-the-maths-drip)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip --math LoRA as a low-rank weight update, and why its residual is the ε-space interaction term
from layer 8

Background
This is layer 12 of the composition ladder in artifacts/drips/simple-distributions-rebuild/: the
correction at scale. poe_repair_min's correction is a LoRA on SDXL, and this drip shows it is a
learned version of layer 8's interaction term. The maths comes before the build and teaches only
what the build needs. I have passed quiz 12.1; any question it marked missed becomes a first rung.

Objectives
1. Teach a LoRA as a frozen projection plus a trained rank-r update, and count its parameters.
2. Derive the guided residual w(ε_J − ε_A − ε_B + ε_∅) from the two guided predictions.
3. Show that its bracket is layer 8's ε-space interaction term.
4. Work a rank-1 update to a 2 × 2 weight matrix on the whiteboard.

What to read
- The layer 8 --math drip folder under artifacts/drips/ and its dispatch.md: my notation for r(x)
  and ε_joint − ε_A − ε_B + ε_∅.
- Quiz 12.1's results: the misses that become first rungs.
- The journey plan 08-lora-update.md, linked on layer 12's Study first line: the parameter count.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- A LoRA keeps W frozen and trains a rank-r update BA, (W + BA)x, with r·(in + out) parameters: 384
  for the journey's toy (rank 2, 3 blocks) and 4,956,160 for the real case (rank 8, 210 modules)
  (the journey plan 08-lora-update.md).
- poe_repair_min's LoRA is rank-8 cross-attention, trained on the cached guided residual
  r_t = ε̃_J − ε̃_PoE (its scope 04 master plan, Mission).
- With ε̃_PoE = ε_∅ + w(ε_A − ε_∅) + w(ε_B − ε_∅) and ε̃_J = ε_∅ + w(ε_J − ε_∅), the residual is
  w(ε_J − ε_A − ε_B + ε_∅), the cache's gs*(eps_j − eps_a − eps_b + eps_uncond) at w = 7.5 (the
  testbed idea map, claim 1; interaction_bound.json, definition).
- The bracket is layer 8's ε-space interaction term (REBUILD_WALK.md, brick "The interaction term,
  exact").
- A rank-1 update to a 2 × 2 weight matrix is the whiteboard instance (layer 12's Achieve line).

Rules
- One rung per message, in LaTeX.
- Start from quiz 12.1's misses.

What to hand back
- The drip's thread folder under artifacts/drips/, with dispatch.md recording my notation, the
  worked numbers I used and what clicked.
- The folder's path, which quiz 12.2 and step 9's build read.

What you will see
Along the way: one rung per message, each ending on the drip's control line.
At the end, a closing report: what was asked, in one sentence; what was done, with the thread
folder's path; PASS or MISS against the target below; what is still unclear to me; at most one
question.

I judge, without notes: done when I derive w(ε_J − ε_A − ε_B + ε_∅) from the two guided predictions and count a LoRA's parameters for one projection by hand.
```

added: the seven sections, from the field-labelled version · "one rung per message, in LaTeX" as a rule (from the layer 1 maths drip's rules, the same skill)
dropped: nothing
