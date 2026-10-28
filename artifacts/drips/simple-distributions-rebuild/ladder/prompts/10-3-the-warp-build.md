# Layer 10, step 3: the warp build

⬅️ [Layer 10, step 3](../10-latent-space-and-vaes.md#3-you-the-build)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip-execute-plan --request compose before and after a warp on the quadrant toy, reproducing
artifacts/ideas/when-poe-reaches-a-and-b/runs/compose-before-or-after-a-warp/, I type every block
myself into ~/Playground/PhD/simple-distributions-walk/walk10.py; bar: difference 4e-17 for
construction (1), 0.126 for the plain product at area stretch 49

Background
This is layer 10's build in the composition ladder (artifacts/drips/simple-distributions-rebuild/).
I compose on the quadrant toy in a latent z, push the result through an invertible warp, and compare
it with composing in pixel space after the warp. The reference is the warp check in the run folder
named above; no rebuild/ module exists for this layer. The setup's folder and environment are in
place, and the layer 10 maths drip folder under artifacts/drips/ holds my notation in its
dispatch.md (step 2).

Objectives
1. Cut the warp check into 4 to 6 pieces, write the step file at the open, and build, teach and test
   one piece per round.
2. I type every block into ~/Playground/PhD/simple-distributions-walk/walk10.py and run it from that
   folder with /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python.
3. walk10.py prints, per warp strength, det J's range and the three total variations (construction
   (1), negation, the plain product).
4. End with a figure of the plain product composed before and after the warp at a = 3, side by side.

What to read
- artifacts/ideas/when-poe-reaches-a-and-b/runs/compose-before-or-after-a-warp/warp_check.py: the
  reference code.
- The same folder's warp_check.json, run.log and README.md: the reference numbers and the run card.
- The layer 10 maths drip's dispatch.md: my notation.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The reference is the quadrant toy in a latent z, one Gaussian blob per quadrant at (±0.5, ±0.5)
  with std 0.15, A∩B share r = 0.05 on the top-right blob and (1 − r)/3 on each other blob
  (warp_check.py).
- Expert A is the right half's data (top-right and bottom-right weights) and expert B the top half's
  (top-right and top-left); the unconditional p is all four blobs (warp_check.py).
- Everything is evaluated on one 801 × 801 latent grid over [−1.6, 1.6]² (warp_check.py).
- A pixel-space expert is the pushforward p_Z(z)/|det J|, and a cell's mass is p_X(x)·|det J|·dz, so
  both routes are compared as probability over the same cells by total variation (warp_check.py;
  the run card).
- The warp is x1 = z1 + a·tanh(2 z1), x2 = z2 + a·tanh(2 z2) + 0.4 z1 at a = 0, 0.3, 1 and 3
  (warp_check.json).
- det J's smallest and largest values are 1 and 1, 1.008 and 2.56, 1.027 and 9, and 1.081 and 49 at
  those four strengths (warp_check.json).
- Negation uses w = 0.3, so p_A^1.3 / p_B^0.3 (warp_check.py).
- Construction (1) differs by 3.72e-17, 3.71e-17 and 3.82e-17 at a = 0.3, 1 and 3, which the card
  rounds to 4e-17 (warp_check.json; run.log).
- Negation differs by 5.0e-17 to 6.6e-17 (warp_check.json; run.log).
- The plain product differs by 0.03539, 0.07957 and 0.1261 (warp_check.json; run.log).
- All three differences are 0 at a = 0 (warp_check.json; run.log).
- The plain product's top-right share barely moves with the warp, 0.9997 in the latent and 0.9998 in
  pixels, so total variation is the number that shows the change, not the share (run.log).
- No rebuild/latent.py exists; the REBUILD_WALK.md brick "Composing in a latent space" names it as
  not built, so the run folder is the only reference (REBUILD_WALK.md).

Rules
- I type; nothing is pasted into walk10.py.
- Never edit the run folder or rebuild/.
- No commit.
- The first piece prints the latent mixture's mass summing to 1, the warp piece prints det J's range
  per a, and each construction's piece prints its four total variations; show each before moving
  on.

What to hand back
- walk10.py, one script that prints, per warp strength, det J's range and the three total
  variations.
- The step file and walk map the --request open writes, with each piece ticked.
- The figure of the plain product composed before and after the warp at a = 3.
- The retrospective the walk closes on, and the /drip-retro command it prints.
- The proof: python walk10.py prints four rows matching run.log in the run folder, tv_plain_product
  0, 0.03539, 0.07957, 0.1261, and tv_construction_1 under 1e-15.

What you will see
Along the way: one piece per message, each ending on the number it printed and a yes to move on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk10.py's path
and the figure's; PASS or MISS against 0.1261 (to four digits) for the plain product at a = 3 and a
construction (1) difference under 1e-15 at every a; what is still open; at most one question.

I judge, from the printed lines: done when the plain product reads 0.1261 at a = 3 and construction (1) differs by under 1e-15 at every a.
```

added: the seven sections, from the field-labelled version · the build's purpose in Background (from the layer 10 page's Why it is critical line)
dropped: nothing
