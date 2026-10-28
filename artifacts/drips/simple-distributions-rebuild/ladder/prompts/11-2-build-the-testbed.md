# Layer 11, step 2: build the testbed

⬅️ [Layer 11, step 2](../11-the-text-conditioned-testbed.md#2-you-build-the-testbed)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`,
once step 1's walk has compiled, with the plan file it names in place of the placeholder.

```text
/drip-execute-plan <the plan file step 1's compile names>

Background
Layer 11 of the composition ladder (artifacts/drips/simple-distributions-rebuild/) is a 32 × 32
text-conditioned testbed, built so each candidate cause of SDXL's cat-dog hybrid can be switched on
one at a time. Step 1's idea walk settled the candidate causes, their signatures and the parser's
categories in artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, and its compile wrote the plan
named above and the route 8 return. The build runs in three stages on the same 32 × 32 data, and
each stage is compared only with the one before it. Layer 10 comes before this one on the ladder's
critical path.

Objectives
1. Walk the compiled plan one step per message in its order.
2. Check the parser bar and the timing bar first, before any stage is compared.
3. Stage A, a text encoder: labels become learned embeddings that condition the denoiser by
   addition, with no attention; against layer 10, only the conditioning changes.
4. Stage B, cross-attention: the same embeddings enter through cross-attention instead; against
   stage A, only the route the text takes changes.
5. Stage C, two encoders: a second encoder's embedding is joined to the first, as in SDXL; against
   stage B, only the second encoder is new.
6. In each stage, test the candidate causes step 1's walk assigned to it, against the bars its plan
   sets.
7. I type every block myself into ~/Playground/PhD/simple-distributions-walk/walk11.py, and into any
   module the plan names in the same folder, and run it with
   /home-mscluster/mmolefe/miniforge3/envs/co3/bin/python.

What to read
- The compiled plan step 1 names: the steps, their order and the bars.
- artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md: the candidate causes, signatures, parser
  categories and Checks outstanding.
- artifacts/ideas/when-poe-reaches-a-and-b/routes/08-sdxl-shaped-hybrid-testbed.return.md: the
  route 8 return.
- artifacts/drips/simple-distributions-rebuild/ladder/11-the-text-conditioned-testbed.md, "Three
  builds, one variable each": the three stages.
- artifacts/drips/simple-distributions-rebuild/REBUILD_WALK.md, brick "The text-conditioned
  testbed".

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The candidate causes, their signatures and the parser's categories are the ones step 1 settled in
  artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md (step 1).
- The parser is trusted only above 99% correct on 1,000 held-out training draws and on 200
  hand-built hybrids (testbed map, Checks outstanding, claim 3).
- The smallest model on single-concept captions should reach 95% parser-correct single-concept
  samples in under 30 minutes, and over an hour means shrink the UNet or the vocabulary first
  (testbed map, Checks outstanding, claim 6).
- Pixel space has no exact density, so success is scored in the drawing parameters (REBUILD_WALK.md,
  brick "The text-conditioned testbed").
- Train on a card with under 1 GB in use, else wait (the GPU guard in
  ~/Playground/PhD/poe_repair_min/environment/hpc/execution-protocol.md).
- The three stages and what each changes against the one before are as the layer 11 page states
  them; step 1's walk picks the causes each stage tests and its plan sets the bars (the layer 11
  page, "Three builds, one variable each").

Rules
- I type; nothing is pasted into walk11.py or the plan's modules.
- Never edit rebuild/ or the idea maps beyond the ticks the plan names.
- No commit.
- The parser check prints its accuracy on each set, the timed run prints minutes to 95%, and each
  later step prints the number the plan names; show each before moving on.
- Compare each stage only with the one before it.

What to hand back
- walk11.py and the modules the plan names.
- The plan's steps ticked, and the walk map that holds my place between sittings.
- The retrospective the walk closes on, and the /drip-retro command it prints.
- The proof: the command the plan names for its bar, plus python walk11.py printing both parser
  accuracies.

What you will see
Along the way: one plan step per message, each ending on the number it printed and a yes to move
on.
At the end, a closing report: what was asked, in one sentence; what was done, with walk11.py's path
and each module's; PASS or MISS against the parser bar (above 99% on both sets), the timing bar
(95% within 30 minutes) and the plan's own bar for each stage; what is still open; at most one
question.

I judge, from the printed lines: done when the parser clears 99% on both sets, the timing run reaches 95% within 30 minutes, and each stage meets the bar step 1's plan sets.
```

added: the seven sections, from the field-labelled version · the objectives split into the three stages A, B and C, each with what it changes against the one before (from the layer 11 page's "Three builds, one variable each" header; the bars and the causes per stage stay with step 1's plan, nothing invented) · layer 10 on the critical path ahead of this layer (from the layer 10 page's ⭐ title) · "compare each stage only with the one before it" as a rule (from the same header)
dropped: nothing
