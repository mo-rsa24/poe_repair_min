# Layer 6, step 4: the maths drip

⬅️ [Layer 6, step 4](../06-guidance-as-a-dial.md#4-you-the-maths-drip)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
after quiz 6.1.

```text
/drip --math raising a density to a power and what it does to the modes, ending at classifier-free
guidance as p(x)·p(A|x)^w·p(B|x)^w, worked on the quadrant toy's ideal experts

Background
This is layer 6 of the composition ladder in artifacts/drips/simple-distributions-rebuild/: guidance
as a dial. Stable Diffusion's cat-dog hybrid happens at guidance weight 7.5, and every layer so far
ran at 1. The maths comes before the build and teaches only what the build needs: what raising a
density to a power does, and the guided target written exactly. I have passed quiz 6.1; any
question it marked missed becomes a first rung.

Objectives
1. Teach what raising a density to the power w does to its modes, starting from a Gaussian.
2. Derive the guided target p(x)·p(A|x)^w·p(B|x)^w from Bayes' rule.
3. Rewrite it as p(x)^(1 − 2w)·p(x|A)^w·p(x|B)^w and show that at w = 1 it is construction (1).
4. Work it on the four-quadrant toy by hand, ending at Z = r for w = 1.

What to read
- Quiz 6.1's results: the misses that become first rungs.
- artifacts/drips/simple-distributions-rebuild/ladder/06-guidance-as-a-dial.md, its Study first
  line: the journey plans, vault notes, textbook and video for this layer.
- artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md: the toy and its
  w = 1 numbers.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- Raising a Gaussian to the power w multiplies its precision by w, so q_a^(w_a)·q_b^(w_b) is a
  Gaussian with precision w_a/v_a + w_b/v_b while that stays positive (vault note Powers Of
  Gaussians).
- By Bayes' rule p(A|x) = p(x|A)p(A)/p(x), so p(x)·p(A|x)^w·p(B|x)^w is
  p(x)^(1 − 2w)·p(x|A)^w·p(x|B)^w up to a constant, and at w = 1 it is construction (1),
  p(x|A)·p(x|B)/p(x) (artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, claim 3).
- SDXL composes two prompts at w = 7.5 as ε_∅ + w(ε_A − ε_∅) + w(ε_B − ε_∅), and the 2D work
  measured only w = 1 (artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, claim 1).
- The quadrant toy here is the four-quadrant answer-key toy, blobs at (±0.5, ±0.5), std 0.15, A is
  x1 > 0 and B is x2 > 0, and at w = 1 with "contains" labels and an A∩B share r it puts 0.989
  within 3 std of (0.5, 0.5) for every r from 0.25 to 0.001, with Z = r
  (artifacts/ideas/when-poe-reaches-a-and-b/runs/answer-key-quadrants/README.md).
- 0.989 is 1 − e^(−9/2), the mass one blob holds within 3 std (when-poe-reaches-a-and-b idea map,
  run 4).
- p(A|x) and p(B|x) are at most 1, so the guided target never exceeds p(x) (Bayes' rule).
- Numbers for w > 1 are the p6. keys step 1's reference writes, so the drip computes its own and
  never quotes one that does not exist yet (layer 6, step 1).

Rules
- One rung per message, in LaTeX, with worked numbers.
- Never quote a w > 1 number the reference has not printed.

What to hand back
- The drip's thread folder under artifacts/drips/, with dispatch.md recording my notation, the
  worked numbers I used and what clicked.
- The folder's path, which quiz 6.2, step 8's scene and step 9's cross-links read.

What you will see
Along the way: one rung per message, each ending on the drip's control line.
At the end, a closing report: what was asked, in one sentence; what was done, with the thread
folder's path; PASS or MISS against the target below; what is still unclear to me; at most one
question.

I judge, without notes: done when I write the guided target from Bayes' rule, rewrite it as p(x)^(1 − 2w)·p(x|A)^w·p(x|B)^w, and show by hand that Z = r at w = 1 on the four-quadrant toy.
```

added: the seven sections, from the field-labelled version · "one rung per message, in LaTeX" as a rule (from the layer 1 maths drip prompt's shape)
dropped: nothing
