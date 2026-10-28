# Layer 11, step 1: the testbed idea walk

⬅️ [Layer 11, step 1](../11-the-text-conditioned-testbed.md#1-you-the-idea-walk)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip-idea resume artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md at claim 1, with
artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md claims 7 and 12 and
artifacts/ideas/poe-overlap-dial/routes/03-what-a-label-teaches-and-the-toy-axes.prompt.md beside
it, until the testbed's candidate causes and its parser are fixed

Background
Layer 11 of the composition ladder (artifacts/drips/simple-distributions-rebuild/) is a 32 × 32
text-conditioned testbed. The 2D toys cannot show failures that come from the text encoder or
cross-attention, such as leakage or a dropped concept, and SDXL has both. The build after this walk
runs in three stages, a text encoder, then cross-attention, then a second encoder, each compared only
with the one before it. This walk picks the candidate causes each stage tests, and its compiled plan
sets the bars. Nothing runs before it in this layer.

Objectives
1. Walk claim 1 of the testbed map, then claims 3 and 5 as far as the parser and the probes need,
   until compile is earned, in repo /home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain.
2. Settle claim 1 with at least two candidate causes, each with a signature computable both in the
   testbed and from the SDXL cache.
3. Fix claim 3's outcome categories.
4. Build the probe table, which places every probe as isolating or not isolating each cause, every
   row naming its check.
5. At compile, write the plan and the route 8 return file at
   artifacts/ideas/when-poe-reaches-a-and-b/routes/08-sdxl-shaped-hybrid-testbed.return.md.

What to read
- artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md, end to end: the testbed map.
- artifacts/ideas/when-poe-reaches-a-and-b/IDEA_MAP.md, end to end: the parent map, claims 7 and 12
  and route 8.
- artifacts/ideas/poe-overlap-dial/routes/03-what-a-label-teaches-and-the-toy-axes.prompt.md, end to
  end: the label route.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- Claim 1 is current, marked wrong as stated: SDXL's cause of the cat-dog hybrid is not known, so
  the testbed cannot inherit it (testbed map, claim 1).
- Claim 1's repair: switch candidate causes one at a time, and a cause becomes a candidate for
  SDXL's hybrid when the signature it leaves in the testbed also appears in SDXL's cache (testbed
  map, claim 1).
- SDXL's hybrid comes from one model with two prompts at guidance 7.5 over 50 DDIM steps, composed
  as ε_∅ + w(ε_A − ε_∅) + w(ε_B − ε_∅), which targets p(x)·p(A|x)^w·p(B|x)^w (testbed map, claim 1
  and Sources).
- That composition is inferred from the cache's residual formula gs*(eps_j − eps_a − eps_b +
  eps_uncond), and the 2D walk measured construction (1) only at w = 1 (testbed map, claim 1 and
  Sources).
- The cache of ε_A, ε_B, ε_joint and ε_∅ at all 50 steps is under
  /datasets/mmolefe/poe_repair_min/artifacts/caches/training_cache/ (testbed map, claim 1 and
  Sources).
- The lower bound on the interaction computed from that cache for 25 animal pairs did not track
  which pairs blend (testbed map, claim 1 and Sources).
- SDXL's hybrids are scored by an instance counter that cannot tell two dogs from a cat and a dog
  (testbed map, Sources: poe_repair_min/context/world/chimera.md).
- Claim 2: total variation to the exact product needs the composed density, which pixel space does
  not give; its repair is a parser that maps each output back to its drawing parameters, where the
  generator's density is known exactly (testbed map, claim 2).
- Claim 3's check: fix the outcome categories (fusion, dominance, leakage among them) before any
  model trains (testbed map, claim 3 and Checks outstanding).
- Claim 3's check then runs the parser on 1,000 held-out training draws and on 200 hand-built
  hybrids (a red dot cluster, a blue circle, a circle made of dots), and above 99% correct on both
  makes it the scorer (testbed map, claim 3 and Checks outstanding).
- Claim 4: a jointly trained tiny encoder has no pressure to put two words close; its repair is a
  frozen embedding table with the two words' cosine set by hand (testbed map, claim 4).
- Claim 5: an attention map is an observation, so every probe is paired with an intervention (mask
  one word's attention to its region, swap the two maps, drop cross-attention at one resolution)
  (testbed map, claim 5).
- Claim 6's check: train the smallest model on single-concept captions in co3 on one GPU until the
  parser calls 95% of single-concept samples correct (testbed map, Checks outstanding).
- Under 30 minutes for that run means dials run as parallel jobs, and over an hour means shrink the
  UNet or the vocabulary (testbed map, Checks outstanding).
- Claim 7 of the parent map: a better network only gets closer to its target density, and region
  and box conditioning change the target (parent map, claim 7).
- Claim 7 also says GLIDE composed as in Liu et al. 2022 is construction (1) on a large base, and
  what an image model adds that 2D cannot show is inductive bias across image parts (parent map,
  claim 7).
- Claim 12 of the parent map needs a check: train on single-concept captions, ask for two concepts
  as the joint caption, construction (1) and the plain product, and score with a parser (parent
  map, claim 12).
- Claim 12's dials are embedding similarity, data overlap and fine-tuning set size, with each
  token's cross-attention map read; Okawa et al. 2023 is confident and Park et al. 2024 is "likely,
  verify" (parent map, claim 12).
- Route 8, which launched this walk, asks for a return at
  artifacts/ideas/when-poe-reaches-a-and-b/routes/08-sdxl-shaped-hybrid-testbed.return.md, and no
  return exists yet (parent map, route 8).
- The testbed must first reproduce that construction (1) with "contains" labels holds 0.989 in A∩B
  down to a data share of 0.001 (route 8's prompt; parent map claims 2 and 9, runs 1 and 5).
- Separately fine-tuned experts reach at most 0.5 when each one's leftover density sits in the two
  adjacent quadrants, which is the answer key's case; leftover on the top-right alone gives 0.9997,
  ladder layer 5's corrected answer key (route 8's prompt; parent map claims 2 and 9, runs 1 and 5).
- Experts over their own part reach 0.9999 with no A∩B data (route 8's prompt; parent map claims 2
  and 9, runs 1 and 5).
- Reverse diffusion with summed scores misses by 0.049 total variation above the exact-sample floor
  (route 8's prompt; parent map claims 2 and 9, runs 1 and 5).
- Chimera mass is 0.66 when both experts saw their object in the same place (route 8's prompt;
  parent map claims 2 and 9, runs 1 and 5).
- The label route proposes that what a product produces is decided by what each label meant in
  training: "contains A" puts 1.00 of the composed mass in A∩B and "exactly A" puts 0.08 (the
  overlap-dial map, claim 1.2).
- The label route was never returned, and the overlap-dial map is closed with its claim 8 already
  taken by the 2D-toys claim, so its axes and prediction are read here as material for claim 1, not
  written back there (the route prompt; the overlap-dial map's header).
- The work uses Python, NumPy and PyTorch only, the co3 and co3_bw environments, and every
  experiment small enough to run in minutes to an hour (route 8's prompt).
- The walk's time box is two sittings of about 90 minutes (testbed map, "The bar and the time box").

Rules
- Each round closes one part with a mark: holds, needs a check, wrong as stated with its repair, or
  dead.
- Write only in the testbed map's folder and the route 8 return file.
- Write nothing in the parent map or the overlap-dial map.
- Train nothing during the walk; claim 6's timing run belongs to step 2.
- No commit.

What to hand back
- The testbed map with claim 1 marked settled, the candidate causes and their signatures in it,
  claim 3's outcome categories fixed, and the probe table.
- At compile, the plan the compile writes and the route 8 return file.
- The compiled plan's path and the route 8 return, which step 2 reads.
- The proof: grep -n "^| \*\*1\.\|^| 1\." artifacts/ideas/sdxl-shaped-hybrid-testbed/IDEA_MAP.md
  prints claim 1's row with its settled mark.

What you will see
Along the way: one part of a claim per round, each ending on its mark and the control line.
At the end, a closing report: what was asked, in one sentence; what was done, with the map, the
compiled plan and the return file by path; PASS or MISS against claim 1 settled with at least two
candidate causes, each with a signature computable in the testbed and from the SDXL cache, and a
probe table whose every row names its check; what is still open; at most one question.

I judge, against the claim ledger: done when claim 1 is settled with at least two candidate causes, each with a signature computable both in the testbed and from the SDXL cache, and every probe is placed as isolating or not isolating each cause with its check named.
```

added: the seven sections, from the field-labelled version · the three build stages the compiled plan has to serve, in Background (from the layer 11 page's "Three builds, one variable each" header) · the map's bar written out in the closing report and the final line, where the old Target pointed at "the map's own bar above"
dropped: nothing
